"""Tests for the deterministic entity-name fallback in structured_lookup.

Covers the recovery added for queries that name a known entity but whose
vector pass misses the declarative chunk text and whose structured
extraction failed to resolve an entity — e.g. yes/no questions like
"is Katherine Mah an advisor?" (the EAV graph holds her `holds_role`
relation). The pure name matcher is tested directly; the async fallback is
exercised against a real in-memory SQLite database.
"""
from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.knowledge.structured_lookup import (
    entity_name_fallback,
    looks_like_member_roster_question,
    looks_like_source_inventory_question,
    match_entity_labels,
    member_roster_fallback,
    source_inventory_fallback,
)


# ---------------------------------------------------------------------------
# Pure name matcher
# ---------------------------------------------------------------------------

class TestMatchEntityLabels:
    def test_subject_ordered_first_by_position(self):
        # The question's subject comes first; the mentioned company follows.
        candidates = [("a", "Alpinist Studios"), ("b", "Chris Byers")]
        assert match_entity_labels("is Chris Byers the founder of Alpinist Studios", candidates) == (
            "Chris Byers",
            "Alpinist Studios",
        )

    def test_case_insensitive(self):
        candidates = [("a", "Katherine Mah")]
        assert match_entity_labels("is katherine mah an advisor", candidates) == ("Katherine Mah",)

    def test_no_match_returns_empty_tuple(self):
        assert match_entity_labels("who is the ceo", [("a", "Katherine Mah")]) == ()

    def test_ignores_single_char_labels(self):
        # One-char labels would false-positive on common words like "a".
        candidates = [("a", "A")]
        assert match_entity_labels("who is a member of the team", candidates) == ()

    def test_all_matches_returned_in_query_order(self):
        candidates = [("a", "Advisor"), ("b", "Katherine Mah"), ("c", "Chris Byers")]
        assert match_entity_labels("Katherine Mah and Chris Byers", candidates) == (
            "Katherine Mah",
            "Chris Byers",
        )


# ---------------------------------------------------------------------------
# Async fallback against a real SQLite schema
# ---------------------------------------------------------------------------

_ENTITIES = [
    ("kath", "Katherine Mah", "Person"),
    ("chris", "Chris Byers", "Person"),
    ("alpinist", "Alpinist Studios", "Company"),
    ("advisor", "Advisor", "Role"),
]


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE entity (id VARCHAR PRIMARY KEY, label VARCHAR NOT NULL, entity_type VARCHAR NOT NULL, name VARCHAR NOT NULL, created_at DATETIME NOT NULL)"))
        await conn.execute(text("CREATE TABLE attribute (id VARCHAR PRIMARY KEY, namespace VARCHAR NOT NULL, name VARCHAR NOT NULL, value_type VARCHAR NOT NULL, multivalue BOOLEAN NOT NULL)"))
        await conn.execute(text("CREATE TABLE value (id VARCHAR PRIMARY KEY, entity_id VARCHAR NOT NULL, attribute_id VARCHAR NOT NULL, value VARCHAR NOT NULL, searchable BOOLEAN NOT NULL, created_at DATETIME NOT NULL)"))
        await conn.execute(text("CREATE TABLE relation (id VARCHAR PRIMARY KEY, source_entity_id VARCHAR NOT NULL, target_entity_id VARCHAR NOT NULL, relation_type VARCHAR NOT NULL, created_at DATETIME NOT NULL)"))
        await conn.execute(text("CREATE TABLE knowledge_source (source_id VARCHAR PRIMARY KEY, source_name VARCHAR NOT NULL, source_type VARCHAR NOT NULL, is_active BOOLEAN NOT NULL)"))
        await conn.execute(text("INSERT INTO entity (id, label, entity_type, name, created_at) VALUES ('kath', 'Katherine Mah', 'Person', 'Katherine Mah', '2026-01-01'), ('chris', 'Chris Byers', 'Person', 'Chris Byers', '2026-01-01'), ('alpinist', 'Alpinist Studios', 'Company', 'Alpinist Studios', '2026-01-01'), ('advisor', 'Advisor', 'Role', 'Advisor', '2026-01-01')"))
        await conn.execute(text("INSERT INTO attribute (id, namespace, name, value_type, multivalue) VALUES ('attr-web', 'Company', 'website', 'string', 0)"))
        await conn.execute(text("INSERT INTO value (id, entity_id, attribute_id, value, searchable, created_at) VALUES ('v1', 'alpinist', 'attr-web', 'https://alpiniststudios.com/', 1, '2026-01-01')"))
        await conn.execute(text("INSERT INTO relation (id, source_entity_id, target_entity_id, relation_type, created_at) VALUES ('r1', 'kath', 'advisor', 'holds_role', '2026-01-01'), ('r2', 'chris', 'advisor', 'holds_role', '2026-01-01')"))
        await conn.execute(text("INSERT INTO knowledge_source (source_id, source_name, source_type, is_active) VALUES ('s1', 'https://alpiniststudios.com/', 'EXTERNAL_INTEGRATION', 1), ('s2', 'https://alpiniststudios.com/about/', 'EXTERNAL_INTEGRATION', 1), ('s3', 'https://alpiniststudios.com/old/', 'EXTERNAL_INTEGRATION', 0)"))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_returns_relation_facts_for_named_person(session_factory):
    async with session_factory() as session:
        facts = await entity_name_fallback("is Katherine Mah an advisor", session=session)
    assert len(facts) == 1
    fact = facts[0]
    assert fact.entity_label == "Katherine Mah"
    assert fact.relation_type == "holds_role"
    assert fact.value == "Advisor"


@pytest.mark.asyncio
async def test_returns_attribute_facts_for_company(session_factory):
    async with session_factory() as session:
        facts = await entity_name_fallback("what is Alpinist Studios", session=session)
    assert len(facts) == 1
    assert facts[0].attribute == "website"
    assert facts[0].value == "https://alpiniststudios.com/"


@pytest.mark.asyncio
async def test_no_named_entity_returns_empty(session_factory):
    async with session_factory() as session:
        facts = await entity_name_fallback("who is the ceo", session=session)
    assert facts == ()


@pytest.mark.asyncio
async def test_full_name_preferred_over_short_label(session_factory):
    # "Mah" is a substring of "Katherine Mah" but not a candidate on its own;
    # a query naming her full name must resolve to her, not a false match.
    async with session_factory() as session:
        facts = await entity_name_fallback("tell me about Katherine Mah", session=session)
    assert len(facts) == 1
    assert facts[0].entity_label == "Katherine Mah"


@pytest.mark.asyncio
async def test_multiple_entities_subject_facts_first(session_factory):
    # "is Chris Byers the founder of Alpinist Studios" mentions both Chris
    # Byers (subject) and Alpinist Studios (company). Both must resolve,
    # subject facts first, so the subject relation is never trimmed.
    async with session_factory() as session:
        facts = await entity_name_fallback("is Chris Byers the founder of Alpinist Studios", session=session)
    labels = [f.entity_label for f in facts]
    assert "Chris Byers" in labels
    assert "Alpinist Studios" in labels
    assert labels.index("Chris Byers") < labels.index("Alpinist Studios")
    chris = [f for f in facts if f.entity_label == "Chris Byers"]
    assert chris[0].relation_type == "holds_role"
    assert chris[0].value == "Advisor"


# ---------------------------------------------------------------------------
# Source-inventory fallback
# ---------------------------------------------------------------------------

class TestLooksLikeSourceInventoryQuestion:
    def test_inventory_phrasings_match(self):
        for q in (
            "how many links have I ingested",
            "how many sources do you have",
            "what documents were uploaded",
            "did the crawl finish",
            "list the URLs in the knowledge base",
        ):
            assert looks_like_source_inventory_question(q), q

    def test_unrelated_phrasings_do_not_match(self):
        for q in (
            "who is the ceo",
            "is Katherine Mah an advisor",
            "what services do you offer",
            "where is the office",
        ):
            assert not looks_like_source_inventory_question(q), q


@pytest.mark.asyncio
async def test_source_inventory_returns_count_and_active_links(session_factory):
    async with session_factory() as session:
        facts = await source_inventory_fallback("how many links have I ingested", session=session)
    assert len(facts) == 3  # 1 count + 2 active sources (the 3rd is inactive)
    assert facts[0].attribute == "ingested_source_count"
    assert facts[0].value == "2"
    assert {f.value for f in facts[1:]} == {
        "https://alpiniststudios.com/",
        "https://alpiniststudios.com/about/",
    }


@pytest.mark.asyncio
async def test_source_inventory_ignores_non_inventory_query(session_factory):
    async with session_factory() as session:
        facts = await source_inventory_fallback("who is the ceo", session=session)
    assert facts == ()


# ---------------------------------------------------------------------------
# Member-roster fallback
# ---------------------------------------------------------------------------

class TestLooksLikeMemberRosterQuestion:
    def test_roster_phrasings_match(self):
        for q in (
            "who are the members",
            "who is on the team",
            "list the employees",
            "how many staff do you have",
            "who are the people at alpinist",
        ):
            assert looks_like_member_roster_question(q), q

    def test_unrelated_phrasings_do_not_match(self):
        for q in (
            "what services do you offer",
            "is Katherine Mah an advisor",
            "how many links have I ingested",
            "what is alpinist studios",
        ):
            assert not looks_like_member_roster_question(q), q


@pytest.mark.asyncio
async def test_member_roster_returns_people_and_roles(session_factory):
    async with session_factory() as session:
        facts = await member_roster_fallback("who are the members", session=session)
    labels = {f.entity_label for f in facts}
    assert {"Katherine Mah", "Chris Byers"} <= labels
    assert all(f.relation_type == "holds_role" for f in facts)
    assert {f.entity_label: f.value for f in facts}["Katherine Mah"] == "Advisor"


@pytest.mark.asyncio
async def test_member_roster_ignores_non_roster_query(session_factory):
    async with session_factory() as session:
        facts = await member_roster_fallback("what is alpinist studios", session=session)
    assert facts == ()
