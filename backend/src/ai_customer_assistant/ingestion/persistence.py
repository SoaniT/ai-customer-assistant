"""
The I/O boundary for steps 4-6 of ingestion_flow.md: writing chunks, and
resolving/writing the EAV facts an extraction produced for them.

UPDATED: rewritten async, against the real ORM models (note the column is
EmbeddingChunk.text, not chunk_text), and taking chunk_embed's own
EmbeddedChunk objects directly instead of a local ChunkDraft -- process_document
has no reuse-by-checksum concept, so that substitution happens here via the
`reused_embeddings` map (checksum -> embedding) computed by
queue.repository.previous_version_chunk_checksums.
"""

from __future__ import annotations

import hashlib
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    Attribute,
    Entity,
    EmbeddingChunk,
    KnowledgeSourceEntityMap,
    Relation,
    Value,
)
from ingestion.extraction.ontology import (
    RELATION_BACKED_ATTRIBUTES,
    safe_canonicalize_attribute,
    safe_canonicalize_entity_name,
    safe_canonicalize_entity_type,
    safe_canonicalize_relation_type,
    safe_canonicalize_value,
)
from ingestion.pipeline_types import ChunkExtraction


def compute_chunk_checksum(text: str) -> str:
    """Pure: EmbeddedChunk has no checksum field, so we derive one here."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def persist_chunks(
    session: AsyncSession,
    version_id: UUID,
    embedded_chunks: tuple,  # tuple[chunk_embed.types.EmbeddedChunk, ...]
    reused_embeddings: dict[str, tuple[float, ...]],
) -> tuple[str, ...]:
    """Insert every chunk for this version. entity_id starts NULL for all
    of them (step 4); _link_entity_to_chunk sets it once extraction (step 5)
    resolves a real entity. Returns the checksums written, in order.

    Idempotent: ON CONFLICT DO NOTHING on (version_id, chunk_index) means a
    job re-run / double-consumption can never insert duplicate chunk rows --
    the same guarantee the Value/Relation writes have. This is what prevents
    _link_entity_to_chunk from ever seeing two rows for one chunk index."""
    checksums: list[str] = []
    for embedded in embedded_chunks:  # one INSERT per row, needs its own values
        checksum = compute_chunk_checksum(embedded.chunk.text)
        embedding = reused_embeddings.get(checksum, embedded.embedding)
        stmt = (
            pg_insert(EmbeddingChunk)
            .values(
                version_id=version_id,
                entity_id=None,
                chunk_index=embedded.chunk.chunk_index,
                text=embedded.chunk.text,
                embedding=list(embedding),
                page=(embedded.chunk.metadata or {}).get("page"),
                token_count=embedded.chunk.token_count,
                checksum=checksum,
            )
            .on_conflict_do_nothing(
                index_elements=[EmbeddingChunk.version_id, EmbeddingChunk.chunk_index]
            )
        )
        await session.execute(stmt)
        checksums.append(checksum)
    await session.flush()
    return tuple(checksums)


async def _resolve_entity(session: AsyncSession, entity_type: str, name: str) -> UUID:
    """Resolve-or-create an entity, deduplicating as aggressively as the
    schema allows without a migration:

    1. Canonicalize ``entity_type`` through the ingestion ontology, so the
       same real-world type always maps to one canonical string (e.g. both
       ``company`` and ``organization`` -> ``Company``) — this is what stops
       "Alpinist Studios" from being stored once per type label.
    2. Canonicalize ``name`` through the ingestion ontology's entity-name
       synonyms, so name variants ("Alpinist Studio", "Alpinist Studios",
       "alpinist") collapse to ONE canonical display name and one row.
    3. Reuse an existing row whose *canonical* type and case-insensitive
       canonical name match.
    4. Fall back to the ``(entity_type, name)`` upsert — the DB unique
       constraint remains the final guarantee.

    Facts and relations route their entity references through this same
    function, so every synonym spelling resolves to the same entity row."""
    canonical_type = safe_canonicalize_entity_type(entity_type)
    canonical_name = safe_canonicalize_entity_name(name)

    existing = (
        await session.execute(
            select(Entity.id)
            .where(
                func.lower(Entity.name) == canonical_name.lower(),
                Entity.entity_type == canonical_type,
            )
            .order_by(Entity.id)
            .limit(2)
        )
    ).scalars().all()
    if existing:
        # Reuse the oldest row as the canonical identity (case-variant
        # duplicates may still exist from pre-fix data; this stops the
        # write path from creating any new ones).
        return existing[0]

    stmt = (
        pg_insert(Entity)
        .values(label=canonical_name, entity_type=canonical_type, name=canonical_name)
        .on_conflict_do_update(index_elements=[Entity.entity_type, Entity.name], set_={"name": canonical_name})
        .returning(Entity.id)
    )
    return (await session.execute(stmt)).scalar_one()


async def _resolve_attribute(
    session: AsyncSession, namespace: str, name: str, value_type: str, multivalue: bool
) -> UUID:
    """Resolve-or-create an attribute, canonicalizing both its namespace
    (an entity type) and its name, so synonyms collapse to one row (e.g.
    ``organization``/``Company`` namespaces and ``mail``/``email`` names)."""
    canonical_namespace = safe_canonicalize_entity_type(namespace)
    canonical_name = safe_canonicalize_attribute(canonical_namespace, name)
    stmt = (
        pg_insert(Attribute)
        .values(
            namespace=canonical_namespace,
            name=canonical_name,
            value_type=value_type,
            multivalue=multivalue,
        )
        .on_conflict_do_update(
            index_elements=[Attribute.namespace, Attribute.name],
            set_={"value_type": value_type},
        )
        .returning(Attribute.id)
    )
    return (await session.execute(stmt)).scalar_one()


async def _link_entity_to_chunk(session: AsyncSession, version_id: UUID, chunk_index: int, entity_id: UUID) -> None:
    chunk = (
        await session.execute(
            select(EmbeddingChunk).where(
                EmbeddingChunk.version_id == version_id, EmbeddingChunk.chunk_index == chunk_index
            )
        )
    ).scalar_one()
    chunk.entity_id = entity_id
    session.add(KnowledgeSourceEntityMap(version_id=version_id, entity_id=entity_id, relationship_type="DERIVED_CHUNK"))


async def _relation_matches_value(session: AsyncSession, entity_id: UUID, value_text: object) -> bool:
    """True when ``entity_id`` already has a relation to an entity whose name
    matches ``value_text`` (case-insensitive) — i.e. this fact is already
    represented as a graph edge, so a scalar value row would be redundant."""
    if not isinstance(value_text, str) or not value_text.strip():
        return False
    result = await session.execute(
        select(Relation.id)
        .join(Entity, Entity.id == Relation.target_entity_id)
        .where(
            Relation.source_entity_id == entity_id,
            func.lower(Entity.name) == value_text.strip().lower(),
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def _value_matches_relation(session: AsyncSession, source_id: UUID, target_name: str) -> bool:
    """True when ``source_id`` already has a Value on a relation-backed
    attribute whose value matches ``target_name`` — i.e. this fact is already
    represented as a scalar value, so a relation row would be redundant."""
    if not target_name or not target_name.strip():
        return False
    result = await session.execute(
        select(Value.id)
        .join(Attribute, Attribute.id == Value.attribute_id)
        .where(
            Value.entity_id == source_id,
            Attribute.name.in_(RELATION_BACKED_ATTRIBUTES),
            func.lower(Value.value) == target_name.strip().lower(),
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def _persist_one_extraction(session: AsyncSession, version_id: UUID, extraction: ChunkExtraction) -> UUID | None:
    """Write one chunk's resolved entity + facts + relations. Returns the
    resolved entity_id, or None if nothing extractable (step 5's rule)."""
    if extraction.entity is None:
        return None

    entity_type, name = extraction.entity
    entity_id = await _resolve_entity(session, entity_type, name)
    await _link_entity_to_chunk(session, version_id, extraction.chunk_index, entity_id)

    resolved: dict[tuple[str, str], UUID] = {extraction.entity: entity_id}

    async def resolve(entity_type_: str, name_: str) -> UUID:
        key = (entity_type_, name_)
        if key not in resolved:
            resolved[key] = await _resolve_entity(session, entity_type_, name_)
        return resolved[key]

    for fact in extraction.facts:
        canonical_namespace = safe_canonicalize_entity_type(fact.namespace)
        canonical_attribute_name = safe_canonicalize_attribute(canonical_namespace, fact.attribute_name)
        attribute_id = await _resolve_attribute(
            session, fact.namespace, fact.attribute_name, fact.value_type, fact.multivalue
        )
        fact_entity_id = await resolve(fact.entity_type, fact.entity_name)
        # P0 double-storage guard: a relation-backed attribute value ("provides
        # X", "uses X") is redundant once the fact already exists as a graph
        # edge (Entity -> Entity). Whichever representation was written first
        # wins, so the same fact is never stored as both a Value and a
        # Relation.
        if canonical_attribute_name in RELATION_BACKED_ATTRIBUTES and await _relation_matches_value(
            session, fact_entity_id, fact.value
        ):
            continue
        # ON CONFLICT DO NOTHING (unique on entity+attribute+value) makes
        # fact writes idempotent: re-extracting the same fact from another
        # chunk or re-ingesting the document can't create duplicate rows.
        # The value is canonicalized first so synonym values ("company" /
        # "organization") also land on ONE row.
        stmt = (
            pg_insert(Value)
            .values(
                entity_id=fact_entity_id,
                attribute_id=attribute_id,
                value=safe_canonicalize_value(fact.value),
                searchable=fact.searchable,
            )
            .on_conflict_do_nothing(
                index_elements=[Value.entity_id, Value.attribute_id, Value.value]
            )
        )
        await session.execute(stmt)

    for relation in extraction.relations:
        source_id = await resolve(relation.source_entity_type, relation.source_entity_name)
        target_id = await resolve(relation.target_entity_type, relation.target_entity_name)
        # P0 double-storage guard (mirror of the value-side check): a relation
        # is redundant once the source entity already holds a relation-backed
        # attribute value naming the target. Facts are written before
        # relations, so a same-extraction pair deterministically keeps the
        # value and drops the relation.
        if await _value_matches_relation(session, source_id, relation.target_entity_name):
            continue
        # Same idempotency guarantee as facts: a relation (source, target,
        # type) is written at most once, so "Alpinist Studios employs Justin
        # Flores" can't appear twice. The type is canonicalized first so
        # "works for"/"hires"/"employs" all map to ONE relation row.
        stmt = (
            pg_insert(Relation)
            .values(
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation_type=safe_canonicalize_relation_type(relation.relation_type),
            )
            .on_conflict_do_nothing(
                index_elements=[
                    Relation.source_entity_id,
                    Relation.target_entity_id,
                    Relation.relation_type,
                ]
            )
        )
        await session.execute(stmt)

    return entity_id


async def persist_chunk_extractions(
    session: AsyncSession, version_id: UUID, extractions: tuple[ChunkExtraction, ...]
) -> int:
    """Persist every chunk's EAV extraction (steps 5-6). Returns the count
    of distinct entities resolved, for job reporting."""
    resolved_ids = set()
    for extraction in extractions:  # each may write rows depending on prior ones
        entity_id = await _persist_one_extraction(session, version_id, extraction)
        resolved_ids.add(entity_id)
    await session.flush()
    return len(resolved_ids - {None})
