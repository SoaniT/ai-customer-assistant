"""Chat API routes (Phase 5, §4.5).

Exposes the single chat endpoint over the ``ChatService`` instance owned by
the FastAPI app (``app.state.chat_service``, set up in ``main.py``'s
lifespan hook).

``trace_id`` is generated per request at this boundary (§2.1 / §4.6): a new
uuid for every incoming message, propagated to the service so the whole
graph run shares one correlation id. No history is accepted from the caller.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Request

from schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

# Last-line safety net: a chat turn must never surface as an HTTP 500 to the
# customer. The graph and its nodes already degrade gracefully (classifier
# failure, knowledge timeout, etc.), but any residual unexpected exception is
# mapped to the safe fallback reply here.
_SAFE_FALLBACK_REPLY = (
    "Sorry, something went wrong on my end. Thank you for your patience, and please try again later."
)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    service = request.app.state.chat_service
    trace_id = str(uuid.uuid4())
    try:
        reply, citations = await service.handle_message_turn(
            payload.thread_id,
            payload.message,
            trace_id=trace_id,
        )
    except Exception:
        reply, citations = _SAFE_FALLBACK_REPLY, []
    return ChatResponse(
        thread_id=payload.thread_id,
        reply=reply,
        trace_id=trace_id,
        citations=citations,
    )