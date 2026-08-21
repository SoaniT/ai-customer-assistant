"""Orchestration layer: LangGraph node functions for the Supervisor agent.

This is the only module in the package that performs I/O (an LLM call).
It stays thin by design: call the client, parse, decide, return a partial
state update. All actual decisions are delegated to the pure functions in
classification.py and routing.py.
"""
from __future__ import annotations

from typing import Callable

from .classification import parse_llm_response
from .llm_client import SupervisorLLMClient
from .prompt import build_supervisor_system_prompt
from .routing import _SAFE_FALLBACK_RESPONSE, assemble_final_response, decide_route
from .schema import Intent, NextAgent, RequestCategory, SupervisorState


def _unavailable_decision() -> dict:
    """Partial-state update for when classification is impossible (provider
    error, exhausted rate limit, transient outage): terminate the turn with
    the safe fallback instead of crashing the request into an HTTP 500."""
    return {
        "request_category": RequestCategory.OUT_OF_SCOPE.value,
        "domain_confidence": 0.0,
        "intent": Intent.UNKNOWN.value,
        "intent_confidence": 0.0,
        "clarification_required": False,
        "clarification_question": None,
        "clarification_attempts": 0,
        "next_agent": NextAgent.NONE.value,
        "ticket_type": None,
        "final_response": _SAFE_FALLBACK_RESPONSE,
    }


def make_classify_and_route_node(
    llm_client: SupervisorLLMClient,
) -> Callable[[SupervisorState], dict]:
    """Build the classify-and-route node, closing over the LLM client.

    The returned function is what LangGraph invokes on entry: it reads
    user_message / conversation_history, classifies, decides where to route,
    and returns a partial update — never a mutated copy of the input state.

    A classifier failure (provider error, rate limit, outage) degrades to the
    safe-fallback decision rather than raising through the graph into the API
    layer as an HTTP 500.
    """

    def classify_and_route(state: SupervisorState) -> dict:
        try:
            raw_response = llm_client.classify(
                system_prompt=build_supervisor_system_prompt(),
                user_message=state["user_message"],
                conversation_history=state.get("conversation_history", []),
            )
            classification = parse_llm_response(raw_response)
        except Exception:
            return _unavailable_decision()
        decision = decide_route(
            classification=classification,
            prior_attempts=state.get("clarification_attempts", 0),
        )

        return {
            "request_category": classification.request_category,
            "domain_confidence": classification.domain_confidence,
            "intent": classification.intent,
            "intent_confidence": classification.intent_confidence,
            "clarification_required": decision.clarification_required,
            "clarification_question": decision.clarification_question,
            "clarification_attempts": decision.clarification_attempts,
            "next_agent": decision.next_agent,
            "ticket_type": decision.ticket_type,
            "final_response": decision.final_response,
        }

    return classify_and_route


def assemble_response_node(state: SupervisorState) -> dict:
    """Run at the FINALIZE end of the post-downstream conditional edge.

    Reduced to pure string formatting since Phase 3: the routing decision
    (finalize vs. escalate into Ticket Agent) has moved out of this node
    into the ``_route_after_downstream`` conditional edge in ``graph.py``.
    All it does is render ``downstream_result["response"]`` into
    ``final_response`` (with the safe fallback if absent/empty).
    """
    return {"final_response": assemble_final_response(state.get("downstream_result"))}