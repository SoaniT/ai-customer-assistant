"""reconcile legacy EAV data against the ingestion ontology

One-time data reconciliation for rows written before the write-time
canonicalization backstop existed in ingestion/persistence.py:

  1. Entity types/names are canonicalized (``org`` -> ``Company``,
     ``alpinist`` -> ``Alpinist Studios``), then duplicate entities (same
     canonical type + canonical name, case-insensitive) are merged: every
     FK reference is repointed to the oldest row and the rest are deleted.
  2. Attribute namespaces/names are canonicalized and merged the same way
     (``company`` -> ``Company``, ``services`` -> ``service``).
  3. Relation types are canonicalized so free-text variants ("formerly
     employed at", "initiated", "developed by") fold onto one spelling.
  4. Values are canonicalized and collapsed (case-variant/synonym values
     land on one row, respecting the unique (entity, attribute, value)).

Deletion/update order guarantees the existing unique constraints
(uq_entity_type_name, uq_attribute_namespace_name, uq_value_*,
uq_relation_*) are never violated mid-migration: duplicate rows are removed
first (lowest id survives), then the survivors are updated to their final
canonical values.

Run after 2a1c9f0e45d7. Idempotent -- a second run finds nothing to merge.

Revision ID: b4e2d9a1c7f3
Revises: 2a1c9f0e45d7
Create Date: 2026-08-17 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "b4e2d9a1c7f3"
down_revision: Union[str, None] = "2a1c9f0e45d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _normalize(text_: str) -> str:
    """Lowercase, collapse hyphens/underscores to spaces and repeated
    whitespace -- mirrors ontology._normalize."""
    return " ".join(text_.strip().lower().replace("-", " ").replace("_", " ").split())


def _reconcile(conn) -> None:
    from ingestion.extraction.ontology import (
        safe_canonicalize_attribute,
        safe_canonicalize_entity_name,
        safe_canonicalize_entity_type,
        safe_canonicalize_relation_type,
        safe_canonicalize_value,
    )
    from ingestion.extraction.ontology_data import ATTRIBUTE_SYNONYMS

    # -- 1. Load and canonicalize attributes ---------------------------------
    attrs = conn.execute(
        text("SELECT id, namespace, name, value_type, multivalue FROM attribute ORDER BY id")
    ).fetchall()

    def canonical_attribute(namespace: str, name: str) -> tuple[str, str]:
        ns = safe_canonicalize_entity_type(namespace)
        attr = safe_canonicalize_attribute(ns, name)
        if attr == name:  # not scoped-resolved; try the unconditional map
            attr = ATTRIBUTE_SYNONYMS.get(_normalize(name), name)
        return ns, attr

    attr_final: dict = {a.id: canonical_attribute(a.namespace, a.name) for a in attrs}
    attr_survivor: dict = {}
    for a in sorted(attrs, key=lambda r: r.id):
        key = attr_final[a.id]
        if key in attr_survivor:
            attr_survivor[a.id] = attr_survivor[key]
        else:
            attr_survivor[key] = a.id
            attr_survivor[a.id] = a.id

    # -- 2. Load and canonicalize entities -----------------------------------
    entities = conn.execute(
        text("SELECT id, label, entity_type, name FROM entity ORDER BY id")
    ).fetchall()

    entity_final: dict = {
        e.id: (safe_canonicalize_entity_type(e.entity_type), safe_canonicalize_entity_name(e.name))
        for e in entities
    }
    entity_survivor: dict = {}
    for e in sorted(entities, key=lambda r: r.id):
        key = (entity_final[e.id][0], entity_final[e.id][1].lower())
        if key in entity_survivor:
            entity_survivor[e.id] = entity_survivor[key]
        else:
            entity_survivor[key] = e.id
            entity_survivor[e.id] = e.id

    def final_entity(entity_id):
        return entity_survivor.get(entity_id, entity_id)

    def final_attribute(attribute_id):
        return attr_survivor.get(attribute_id, attribute_id)

    # -- 3. Reconcile values ---------------------------------------------------
    # Collapse onto (final entity, final attribute, canonical value). Duplicates
    # removed first (lowest id), survivors then updated -- the unique constraint
    # can never fire.
    values = conn.execute(
        text("SELECT id, entity_id, attribute_id, value, searchable FROM value ORDER BY id")
    ).fetchall()

    def value_final(v):
        entity_id = final_entity(v.entity_id)
        attribute_id = final_attribute(v.attribute_id)
        value = safe_canonicalize_value(v.value)
        return entity_id, attribute_id, value

    keep_value_ids: set = set()
    for v in sorted(values, key=lambda r: r.id):
        if value_final(v) not in keep_value_ids:
            keep_value_ids.add(value_final(v))
            keep_value_ids.add(v.id)
    for v in values:
        if v.id not in keep_value_ids:
            conn.execute(text("DELETE FROM value WHERE id = :id"), {"id": v.id})
    for v in values:
        if v.id in keep_value_ids:
            final = value_final(v)
            if (v.entity_id, v.attribute_id, v.value) != final:
                conn.execute(
                    text("UPDATE value SET entity_id = :e, attribute_id = :a, value = :v WHERE id = :id"),
                    {"e": final[0], "a": final[1], "v": final[2], "id": v.id},
                )

    # -- 4. Reconcile relations -------------------------------------------------
    # Same pattern: dedupe on (final source, final target, canonical type).
    relations = conn.execute(
        text(
            "SELECT id, source_entity_id, target_entity_id, relation_type FROM relation ORDER BY id"
        )
    ).fetchall()

    def relation_final(r):
        return (
            final_entity(r.source_entity_id),
            final_entity(r.target_entity_id),
            safe_canonicalize_relation_type(r.relation_type),
        )

    keep_relation_ids: set = set()
    for r in sorted(relations, key=lambda x: x.id):
        if relation_final(r) not in keep_relation_ids:
            keep_relation_ids.add(relation_final(r))
            keep_relation_ids.add(r.id)
    for r in relations:
        if r.id not in keep_relation_ids:
            conn.execute(text("DELETE FROM relation WHERE id = :id"), {"id": r.id})
    for r in relations:
        if r.id in keep_relation_ids and (
            final_entity(r.source_entity_id) != r.source_entity_id
            or final_entity(r.target_entity_id) != r.target_entity_id
            or safe_canonicalize_relation_type(r.relation_type) != r.relation_type
        ):
            final = relation_final(r)
            conn.execute(
                text(
                    "UPDATE relation SET source_entity_id = :s, target_entity_id = :t, "
                    "relation_type = :rt WHERE id = :id"
                ),
                {"s": final[0], "t": final[1], "rt": final[2], "id": r.id},
            )

    # -- 5. Repoint derived rows to surviving entities --------------------------
    losers = {e.id for e in entities if entity_survivor.get(e.id) != e.id}
    if losers:
        conn.execute(
            text(
                "UPDATE embedding_chunk SET entity_id = :survivor "
                "WHERE entity_id = :loser"
            ),
            [{"survivor": entity_survivor[loser], "loser": loser} for loser in losers],
        )
        conn.execute(
            text(
                "UPDATE knowledge_source_entity_map SET entity_id = :survivor "
                "WHERE entity_id = :loser"
            ),
            [{"survivor": entity_survivor[loser], "loser": loser} for loser in losers],
        )

    # -- 6. Delete merged attributes/entities, rename survivors ----------------
    for a in attrs:
        if attr_survivor[a.id] != a.id:
            conn.execute(text("DELETE FROM attribute WHERE id = :id"), {"id": a.id})
    for a in attrs:
        if attr_survivor[a.id] == a.id:
            ns, name = attr_final[a.id]
            if (a.namespace, a.name) != (ns, name):
                conn.execute(
                    text("UPDATE attribute SET namespace = :ns, name = :name WHERE id = :id"),
                    {"ns": ns, "name": name, "id": a.id},
                )

    for e in entities:
        if entity_survivor[e.id] != e.id:
            conn.execute(text("DELETE FROM entity WHERE id = :id"), {"id": e.id})
    for e in entities:
        if entity_survivor[e.id] == e.id:
            final_type, final_name = entity_final[e.id]
            if (e.entity_type, e.name) != (final_type, final_name):
                conn.execute(
                    text("UPDATE entity SET label = :name, entity_type = :type, name = :name WHERE id = :id"),
                    {"name": final_name, "type": final_type, "id": e.id},
                )


def upgrade() -> None:
    conn = op.get_bind()
    _reconcile(conn)


def downgrade() -> None:
    # Data reconciliation is intentionally not reversible: merged rows were
    # deleted and cannot be restored. Downgrade is a no-op.
    pass