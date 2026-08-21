"""
structured_lookup.py — deterministic retrieval from the EAV schema.

The only module in the Knowledge Agent package that reads
`entity` / `attribute` / `value` / `relation` / `knowledge_source`
directly — every other module receives its data already resolved. No
vector search, no LLM; pure deterministic reads against the schema
described in schema.md.

Schema note on `entity.label` vs `entity.name`: schema.md defines both
columns, but only `(entity_type, name)` carries the composite-unique
dedup constraint — `name` is therefore the identifying text an
extracted `entity_label` (e.g. "Project Alpha") should match against.
`entity.label`'s further semantics aren't specified beyond its type, so
this module doesn't read or write it; only `entity.name` is surfaced as
`StructuredFact.entity_label` / matched against
`StructuredQuery.entity_label`.

I/O boundary: `_fetch_rows()` is the only place this module calls
`session.execute()`. Everything that builds a `Select` statement is a
pure function — testable by inspecting the statement object without a
database at all; the async round-trip is exercised separately against
a real in-memory SQLite database (aiosqlite), matching the
`ingestion/storage` convention of fakes/sqlite over mocks.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, MetaData, String, Table, func, select
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from .exceptions import AmbiguousEntityError, AttributeNotFoundError, EntityNotFoundError
from .types import StructuredFact, StructuredQuery

# ==========================================================================
# Data layer — structural mirror of schema.md's EAV tables. schema.md
# remains the authoritative source; this is only enough structure to
# build correct Select statements against it.
# ==========================================================================

metadata = MetaData()

entity_table = Table(
    "entity",
    metadata,
    Column("id", String, primary_key=True),
    Column("label", String, nullable=False),
    Column("entity_type", String, nullable=False),
    Column("name", String, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

attribute_table = Table(
    "attribute",
    metadata,
    Column("id", String, primary_key=True),
    Column("namespace", String, nullable=False),
    Column("name", String, nullable=False),
    Column("value_type", String, nullable=False),
    Column("multivalue", Boolean, nullable=False),
)

value_table = Table(
    "value",
    metadata,
    Column("id", String, primary_key=True),
    Column("entity_id", String, ForeignKey("entity.id"), nullable=False),
    Column("attribute_id", String, ForeignKey("attribute.id"), nullable=False),
    Column("value", String, nullable=False),
    Column("searchable", Boolean, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

relation_table = Table(
    "relation",
    metadata,
    Column("id", String, primary_key=True),
    Column("source_entity_id", String, ForeignKey("entity.id"), nullable=False),
    Column("target_entity_id", String, ForeignKey("entity.id"), nullable=False),
    Column("relation_type", String, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

knowledge_source_table = Table(
    "knowledge_source",
    metadata,
    Column("source_id", String, primary_key=True),
    Column("source_name", String, nullable=False),
    Column("source_type", String, nullable=False),
    Column("is_active", Boolean, nullable=False),
)


# ==========================================================================
# Public API
# ==========================================================================


async def entity_name_fallback(query_text: str, *, session: AsyncSession) -> tuple[StructuredFact, ...]:
    """Deterministic last-resort retrieval for queries that name a known
    entity but whose structured extraction failed to resolve one (e.g.
    yes/no questions like "is Katherine Mah an advisor?" — the vector pass
    misses declarative chunk text and extraction often returns no entity).

    Matches EVERY known ``entity.name`` contained in ``query_text``
    (case-insensitive), ordered by first occurrence — the subject of the
    question ("Chris Byers" in "is Chris Byers the founder of Alpinist
    Studios?") precedes merely-mentioned entities — then returns each
    matched entity's attribute values AND all of its outgoing relations
    (``holds_role`` etc.), subject-facts first so the context budget keeps
    them. Empty tuple when no entity is named or no matched entity has
    facts. No LLM, no vector search — pure deterministic EAV reads.
    """
    rows = await _fetch_rows(session, _all_entities_statement())
    matched_names = match_entity_labels(query_text, [(row.id, row.name) for row in rows])
    if not matched_names:
        return ()
    facts: list[StructuredFact] = []
    for name in matched_names:
        row = next((r for r in rows if r.name == name), None)
        if row is None:
            continue
        general = await _general_lookup(
            row,
            StructuredQuery(entity_type=row.entity_type, entity_label=row.name, attribute=None, relation_type=None, filters=(), confidence=0.0),
            session=session,
        )
        relations = await _all_relations_lookup(row, session)
        facts.extend(general)
        facts.extend(relations)
    return tuple(facts)


def match_entity_labels(query_text: str, candidates: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """Pure: every candidate ``(entity_id, name)`` whose ``name`` appears in
    ``query_text`` (case-insensitive), ordered by the position of its first
    occurrence in the query. The subject of a query usually comes first
    ("is Chris Byers the founder of Alpinist Studios"), so position-order
    keeps the subject's facts ahead of merely-mentioned entities' facts.
    Returns an empty tuple when no candidate name is mentioned. Entities
    with one-character names are ignored (they false-positive on common
    words like "a")."""
    lowered = query_text.lower()
    matches = [
        (lowered.find(name.lower()), name)
        for _, name in candidates
        if name and len(name) >= 2 and name.lower() in lowered
    ]
    matches.sort()
    return tuple(name for _, name in matches)


# --------------------------------------------------------------------------
# Source-inventory fallback (knowledge_source reads)
# --------------------------------------------------------------------------

_SOURCE_INVENTORY_KEYWORDS = ("source", "link", "ingest", "crawl", "upload", "document", "url")


def looks_like_source_inventory_question(query_text: str) -> bool:
    """Pure: whether ``query_text`` reads like a question about the
    ingested source inventory ("how many links have I ingested?", "what
    sources do you have?"). Matches on a handful of inventory keywords;
    used as a last-resort gate so the deterministic count below only
    runs when retrieval has already come up empty."""
    lowered = query_text.lower()
    return any(kw in lowered for kw in _SOURCE_INVENTORY_KEYWORDS)


def _source_inventory_statement() -> Select:
    """Pure: every active knowledge source's id and name, for the
    inventory facts. `source_id` is a UUID — the fact's entity_id accepts
    any string — and `source_name` carries the URL/path the user knows."""
    return select(
        knowledge_source_table.c.source_id,
        knowledge_source_table.c.source_name,
    ).where(knowledge_source_table.c.is_active.is_(True))


async def source_inventory_fallback(query_text: str, *, session: AsyncSession) -> tuple[StructuredFact, ...]:
    """Deterministic last-resort for source-inventory questions ("how many
    links have I ingested?", "how many sources do you have?"). The count
    and the URLs live ONLY in ``knowledge_source`` — never in the chunk
    corpus or the EAV graph — so no vector/entity fallback can answer them.
    Returns the active source count plus one fact per active source, or ()
    when the query isn't inventory-flavored or no active sources exist. No
    LLM, no vector search — pure deterministic reads.
    """
    if not looks_like_source_inventory_question(query_text):
        return ()
    rows = await _fetch_rows(session, _source_inventory_statement())
    if not rows:
        return ()
    facts: list[StructuredFact] = [
        StructuredFact(
            entity_id="knowledge-base",
            entity_type="KnowledgeBase",
            entity_label="Knowledge Base",
            attribute="ingested_source_count",
            value=str(len(rows)),
            value_type="number",
            confidence=1.0,
        )
    ]
    facts.extend(
        StructuredFact(
            entity_id=row.source_id,
            entity_type="KnowledgeSource",
            entity_label=row.source_name,
            attribute="ingested_link",
            value=row.source_name,
            value_type="string",
            confidence=1.0,
        )
        for row in rows
    )
    return tuple(facts)


# --------------------------------------------------------------------------
# Member-roster fallback (Person entity reads)
# --------------------------------------------------------------------------

_MEMBER_KEYWORDS = (
    "member",
    "team",
    "employee",
    "staff",
    "people",
    "colleague",
    "who works",
    "who is at",
    "who is part",
    "roster",
    "org chart",
)


def looks_like_member_roster_question(query_text: str) -> bool:
    """Pure: whether ``query_text`` asks to enumerate the company's people
    ("who are the members?", "list the team"). Used as a last-resort gate:
    such phrasings embed too far from the declarative team list to vector-hit,
    but the EAV graph holds every Person and their ``holds_role`` relations."""
    lowered = query_text.lower()
    return any(kw in lowered for kw in _MEMBER_KEYWORDS)


def _member_roster_statement() -> Select:
    """Pure: every Person entity, for roster enumeration."""
    return select(entity_table.c.id, entity_table.c.entity_type, entity_table.c.name).where(
        entity_table.c.entity_type == "Person"
    )


async def member_roster_fallback(query_text: str, *, session: AsyncSession) -> tuple[StructuredFact, ...]:
    """Deterministic last-resort for "who are the members/team" questions.
    Returns the outgoing relations (``holds_role`` etc.) of every Person
    entity, so the answer LLM can list people and their positions. Empty
    tuple when the query isn't roster-flavored or there are no Person
    entities. No LLM, no vector search — pure deterministic EAV reads.
    """
    if not looks_like_member_roster_question(query_text):
        return ()
    persons = await _fetch_rows(session, _member_roster_statement())
    if not persons:
        return ()
    facts: list[StructuredFact] = []
    for person in persons:
        facts.extend(await _all_relations_lookup(person, session))
    return tuple(facts)


async def structured_lookup(query: StructuredQuery, *, session: AsyncSession) -> tuple[StructuredFact, ...]:
    """Deterministic EAV lookup for a canonicalized StructuredQuery.

    Returns an empty tuple (never None) when `query.entity_type` is
    unset — there's nothing structured to look up, which is a valid
    outcome, not a failure. Also returns an empty tuple when the
    resolved entity has no matching values/relations for a *general*
    lookup (no specific attribute or relation was requested).

    Raises EntityNotFoundError if no entity matches (entity_type,
    entity_label). Raises AmbiguousEntityError if more than one does.
    Raises AttributeNotFoundError if a *specific* attribute was
    requested and the resolved entity has no value for it."""
    if query.entity_type is None:
        return ()

    entity_row = await _resolve_single_entity(query, session=session)
    lookup_kind = _lookup_kind(query)
    handler = _LOOKUP_DISPATCH[lookup_kind]
    return await handler(entity_row, query, session)


# ==========================================================================
# Internals — entity resolution
# ==========================================================================


def _entity_lookup_statement(query: StructuredQuery) -> Select:
    """Pure: build (never execute) the entity-resolution statement."""
    stmt = select(entity_table.c.id, entity_table.c.entity_type, entity_table.c.name).where(
        entity_table.c.entity_type == query.entity_type
    )
    if query.entity_label is not None:
        stmt = stmt.where(func.lower(entity_table.c.name) == query.entity_label.lower())
    return stmt


def _all_entities_statement() -> Select:
    """Pure: every known entity's id/type/name, for name matching."""
    return select(entity_table.c.id, entity_table.c.entity_type, entity_table.c.name)


async def _resolve_single_entity(query: StructuredQuery, *, session: AsyncSession) -> Row:
    rows = await _fetch_rows(session, _entity_lookup_statement(query))
    if len(rows) == 0:
        raise EntityNotFoundError(
            message=f"no entity found for entity_type={query.entity_type!r}, entity_label={query.entity_label!r}",
            entity_type=query.entity_type,
            entity_label=query.entity_label or "",
        )
    if len(rows) > 1:
        raise AmbiguousEntityError(
            message=(
                f"{len(rows)} entities match entity_type={query.entity_type!r}, "
                f"entity_label={query.entity_label!r} — need a more specific label"
            ),
            entity_type=query.entity_type,
            entity_label=query.entity_label or "",
            candidate_count=len(rows),
        )
    (single_row,) = rows
    return single_row


# ==========================================================================
# Internals — lookup-kind dispatch (relation / attribute / general)
# ==========================================================================


def _lookup_kind(query: StructuredQuery) -> str:
    """Pure: which of the three lookup shapes applies. A relation
    request takes priority over an attribute request since extraction.py
    only ever populates one of the two for a given query."""
    return next(
        kind
        for kind, is_applicable in (
            ("relation", query.relation_type is not None),
            ("attribute", query.attribute is not None),
            ("general", True),
        )
        if is_applicable
    )


def _attribute_lookup_statement(entity_id: str, entity_type: str, attribute_name: str) -> Select:
    return (
        select(value_table.c.value, attribute_table.c.value_type)
        .select_from(value_table.join(attribute_table, value_table.c.attribute_id == attribute_table.c.id))
        .where(value_table.c.entity_id == entity_id)
        .where(attribute_table.c.namespace == entity_type)
        .where(attribute_table.c.name == attribute_name)
    )


async def _attribute_lookup(entity_row: Row, query: StructuredQuery, session: AsyncSession) -> tuple[StructuredFact, ...]:
    rows = await _fetch_rows(
        session, _attribute_lookup_statement(entity_row.id, entity_row.entity_type, query.attribute)
    )
    if not rows:
        raise AttributeNotFoundError(
            message=(
                f"entity {entity_row.name!r} ({entity_row.entity_type}) has no value "
                f"for attribute {query.attribute!r}"
            ),
            entity_type=entity_row.entity_type,
            entity_label=entity_row.name,
            attribute=query.attribute,
        )
    return tuple(
        StructuredFact(
            entity_id=entity_row.id,
            entity_type=entity_row.entity_type,
            entity_label=entity_row.name,
            attribute=query.attribute,
            value=row.value,
            value_type=row.value_type,
        )
        for row in rows
    )


def _general_values_statement(entity_id: str) -> Select:
    return (
        select(attribute_table.c.name.label("attribute_name"), value_table.c.value, attribute_table.c.value_type)
        .select_from(value_table.join(attribute_table, value_table.c.attribute_id == attribute_table.c.id))
        .where(value_table.c.entity_id == entity_id)
    )


async def _general_lookup(entity_row: Row, query: StructuredQuery, session: AsyncSession) -> tuple[StructuredFact, ...]:
    rows = await _fetch_rows(session, _general_values_statement(entity_row.id))
    return tuple(
        StructuredFact(
            entity_id=entity_row.id,
            entity_type=entity_row.entity_type,
            entity_label=entity_row.name,
            attribute=row.attribute_name,
            value=row.value,
            value_type=row.value_type,
        )
        for row in rows
    )


def _relation_lookup_statement(entity_id: str, relation_type: str) -> Select:
    target = entity_table.alias("target_entity")
    return (
        select(
            relation_table.c.relation_type,
            target.c.name.label("target_name"),
            target.c.entity_type.label("target_type"),
        )
        .select_from(relation_table.join(target, relation_table.c.target_entity_id == target.c.id))
        .where(relation_table.c.source_entity_id == entity_id)
        .where(relation_table.c.relation_type == relation_type)
    )


async def _relation_lookup(entity_row: Row, query: StructuredQuery, session: AsyncSession) -> tuple[StructuredFact, ...]:
    rows = await _fetch_rows(session, _relation_lookup_statement(entity_row.id, query.relation_type))
    return tuple(
        StructuredFact(
            entity_id=entity_row.id,
            entity_type=entity_row.entity_type,
            entity_label=entity_row.name,
            attribute=row.relation_type,
            value=row.target_name,
            value_type="string",
            related_entity_label=row.target_name,
            relation_type=row.relation_type,
        )
        for row in rows
    )


def _all_relations_statement(entity_id: str) -> Select:
    target = entity_table.alias("target_entity")
    return (
        select(
            relation_table.c.relation_type,
            target.c.name.label("target_name"),
            target.c.entity_type.label("target_type"),
        )
        .select_from(relation_table.join(target, relation_table.c.target_entity_id == target.c.id))
        .where(relation_table.c.source_entity_id == entity_id)
    )


async def _all_relations_lookup(entity_row: Row, session: AsyncSession) -> tuple[StructuredFact, ...]:
    rows = await _fetch_rows(session, _all_relations_statement(entity_row.id))
    return tuple(
        StructuredFact(
            entity_id=entity_row.id,
            entity_type=entity_row.entity_type,
            entity_label=entity_row.name,
            attribute=row.relation_type,
            value=row.target_name,
            value_type="string",
            related_entity_label=row.target_name,
            relation_type=row.relation_type,
        )
        for row in rows
    )


_LOOKUP_DISPATCH: Mapping[str, Callable] = {
    "relation": _relation_lookup,
    "attribute": _attribute_lookup,
    "general": _general_lookup,
}


# ==========================================================================
# Internals — I/O boundary (the only place this module executes SQL)
# ==========================================================================


async def _fetch_rows(session: AsyncSession, stmt: Select) -> Sequence[Row]:
    result = await session.execute(stmt)
    return result.all()