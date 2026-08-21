"""Regression test: `structured_facts` must tolerate concurrent writers.

The hybrid retrieval strategy fans out to `structured_lookup` and
`vector_search` in parallel. The vector-search node also emits
`structured_facts` (the entity-name fallback), so both branches write
the same channel in one superstep. With a plain `LastValue` channel that
raises `InvalidUpdateError: At key 'structured_facts': Can receive only
one value per step`; `structured_facts` is therefore declared as an
`Annotated[tuple, operator.add]` reducer channel in state.py. This test
proves the channel merges concurrent writes from a real compiled graph.
"""
from __future__ import annotations

import pytest
from langgraph.graph import END, StateGraph

from agents.knowledge.state import KnowledgeAgentState
from agents.knowledge.types import StructuredFact


async def _write_facts_factory(entity_label: str, value: str):
    async def _node(state: KnowledgeAgentState) -> dict:
        fact = StructuredFact(
            entity_id=f"id-{entity_label}",
            entity_type="Person",
            entity_label=entity_label,
            attribute="holds_role",
            value=value,
            value_type="string",
            related_entity_label=value,
            relation_type="holds_role",
        )
        return {"structured_facts": (fact,)}

    return _node


@pytest.mark.asyncio
async def test_structured_facts_merges_concurrent_writes():
    async def _route(state: KnowledgeAgentState) -> list[str]:
        # Like decide_strategy's hybrid branch: fan out to both nodes in
        # the SAME superstep, so both write `structured_facts` in parallel.
        return ["structured_lookup", "vector_fallback"]

    g = StateGraph(KnowledgeAgentState)
    g.add_node("structured_lookup", await _write_facts_factory("Katherine Mah", "Advisor"))
    g.add_node("vector_fallback", await _write_facts_factory("Katherine Mah", "Advisor"))
    g.add_conditional_edges("__start__", _route)
    g.add_edge("structured_lookup", END)
    g.add_edge("vector_fallback", END)
    app = g.compile()

    result = await app.ainvoke({"raw_query": "is Katherine Mah an advisor"})

    assert len(result["structured_facts"]) == 2
    assert {f.related_entity_label for f in result["structured_facts"]} == {"Advisor"}