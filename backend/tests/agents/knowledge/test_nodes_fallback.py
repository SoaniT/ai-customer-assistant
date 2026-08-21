"""Node-level tests for the entity-name fallback wiring.

The deterministic name fallback must fire regardless of which strategy
routed to a node: the structured node when EAV lookup comes back empty,
and the vector node when vector retrieval comes back empty. These tests
drive the node factories with faked DB sessions / lookup callables and
assert the returned partial-update dicts (mirroring the pattern described
in nodes.py's module docstring: nodes take injected fakes, no LangGraph
runtime required).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agents.knowledge.config import KnowledgeAgentConfig
from agents.knowledge.exceptions import EmptyRetrievalError, EntityNotFoundError
from agents.knowledge.nodes import make_structured_lookup_node, make_vector_search_node
from agents.knowledge.state import KnowledgeAgentState
from agents.knowledge.types import RewrittenQuery


def _state(*, rewritten: str = "is Katherine Mah an advisor") -> KnowledgeAgentState:
    return KnowledgeAgentState(
        raw_query=rewritten,
        rewritten_query=RewrittenQuery(original_text=rewritten, rewritten_text=rewritten),
    )


class _Session:
    """Bare session stand-in — the lookup functions are monkeypatched, so
    the node's `async with session_factory() as session` only needs to
    yield something."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class TestStructuredLookupNodeFallback:
    def test_structured_miss_merges_all_fallbacks(self, monkeypatch):
        async def fake_lookup(_query, session=None):
            raise EntityNotFoundError("no entity found")

        async def roster(_text, session=None):
            return ("roster-fact",)

        async def named(_text, session=None):
            return ("named-fact",)

        async def inventory(_text, session=None):
            return ("inventory-fact",)

        monkeypatch.setattr("agents.knowledge.nodes.structured_lookup", fake_lookup)
        monkeypatch.setattr("agents.knowledge.nodes.member_roster_fallback", roster)
        monkeypatch.setattr("agents.knowledge.nodes.entity_name_fallback", named)
        monkeypatch.setattr("agents.knowledge.nodes.source_inventory_fallback", inventory)
        node = make_structured_lookup_node(session_factory=lambda: _Session())

        result = asyncio.run(node(_state()))

        # Roster facts first, then named-entity, then inventory — so the
        # intent-specific facts survive the context budget.
        assert result == {"structured_facts": ("roster-fact", "named-fact", "inventory-fact")}

    def test_structured_hit_skips_fallback(self, monkeypatch):
        async def fake_lookup(_query, session=None):
            return ("real-fact",)

        async def boom(_text, session=None):
            raise AssertionError("fallback must not run when EAV lookup hit")

        monkeypatch.setattr("agents.knowledge.nodes.structured_lookup", fake_lookup)
        monkeypatch.setattr("agents.knowledge.nodes.member_roster_fallback", boom)
        monkeypatch.setattr("agents.knowledge.nodes.entity_name_fallback", boom)
        monkeypatch.setattr("agents.knowledge.nodes.source_inventory_fallback", boom)
        node = make_structured_lookup_node(session_factory=lambda: _Session())

        result = asyncio.run(node(_state()))

        assert result == {"structured_facts": ("real-fact",)}

    def test_roster_flavored_query_merges_roster_above_general_lookup(self, monkeypatch):
        # "who are the members of Alpinist Studios?" can fall out of
        # extraction as a GENERAL lookup on the company (relation_type
        # unset) — non-empty, so an empty-only fallback would skip the
        # roster. The intent gate must still run it and put roster facts
        # FIRST so the context budget keeps them.
        async def fake_lookup(_query, session=None):
            return ("company-fact",)

        async def roster(_text, session=None):
            return ("roster-fact",)

        async def named(_text, session=None):
            return ()

        async def inventory(_text, session=None):
            return ()

        monkeypatch.setattr("agents.knowledge.nodes.structured_lookup", fake_lookup)
        monkeypatch.setattr("agents.knowledge.nodes.member_roster_fallback", roster)
        monkeypatch.setattr("agents.knowledge.nodes.entity_name_fallback", named)
        monkeypatch.setattr("agents.knowledge.nodes.source_inventory_fallback", inventory)
        node = make_structured_lookup_node(session_factory=lambda: _Session())

        result = asyncio.run(node(_state(rewritten="who are the members of Alpinist Studios")))

        assert result == {"structured_facts": ("roster-fact", "company-fact")}

    def test_structured_miss_with_empty_fallbacks_returns_empty(self, monkeypatch):
        # EAV empty and no fallback matched -> empty facts (not an error).
        async def fake_lookup(_query, session=None):
            raise EntityNotFoundError("no entity found")

        async def empty(_text, session=None):
            return ()

        monkeypatch.setattr("agents.knowledge.nodes.structured_lookup", fake_lookup)
        monkeypatch.setattr("agents.knowledge.nodes.member_roster_fallback", empty)
        monkeypatch.setattr("agents.knowledge.nodes.entity_name_fallback", empty)
        monkeypatch.setattr("agents.knowledge.nodes.source_inventory_fallback", empty)
        node = make_structured_lookup_node(session_factory=lambda: _Session())

        result = asyncio.run(node(_state()))

        assert result == {"structured_facts": ()}


class TestVectorSearchNodeFallback:
    def test_vector_empty_merges_fallbacks(self, monkeypatch):
        async def fake_search(_query, config=None, session=None, embed_query=None):
            raise EmptyRetrievalError("no chunk met threshold")

        async def roster(_text, session=None):
            return ("roster-fact",)

        async def named(_text, session=None):
            return ("named-fact",)

        async def inventory(_text, session=None):
            return ()

        monkeypatch.setattr("agents.knowledge.nodes.vector_search", fake_search)
        monkeypatch.setattr("agents.knowledge.nodes.member_roster_fallback", roster)
        monkeypatch.setattr("agents.knowledge.nodes.entity_name_fallback", named)
        monkeypatch.setattr("agents.knowledge.nodes.source_inventory_fallback", inventory)
        node = make_vector_search_node(
            config=KnowledgeAgentConfig(embedding_dimension=2),
            session_factory=lambda: _Session(),
            embed_query=lambda _text: (1.0,),
        )

        result = asyncio.run(node(_state()))

        assert result == {"retrieved_chunks": (), "structured_facts": ("roster-fact", "named-fact")}

    def test_vector_hit_skips_fallback(self, monkeypatch):
        async def fake_search(_query, config=None, session=None, embed_query=None):
            return ("chunk",)

        async def boom(_text, session=None):
            raise AssertionError("fallback must not run when vector retrieval hit")

        monkeypatch.setattr("agents.knowledge.nodes.vector_search", fake_search)
        monkeypatch.setattr("agents.knowledge.nodes.member_roster_fallback", boom)
        monkeypatch.setattr("agents.knowledge.nodes.entity_name_fallback", boom)
        monkeypatch.setattr("agents.knowledge.nodes.source_inventory_fallback", boom)
        node = make_vector_search_node(
            config=KnowledgeAgentConfig(embedding_dimension=2),
            session_factory=lambda: _Session(),
            embed_query=lambda _text: (1.0,),
        )

        result = asyncio.run(node(_state()))

        assert result == {"retrieved_chunks": ("chunk",), "structured_facts": ()}
