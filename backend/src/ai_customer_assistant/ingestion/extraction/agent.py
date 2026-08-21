"""
EAV extraction agent (ingestion_flow.md step 5), implemented as a single
JSON-mode completion per chunk -- not a multi-turn agentic loop, since the
task is structured extraction, not multi-step reasoning with external
actions. We deliberately use JSON structured output rather than tool
calling: the models served by Groq that best fit extraction (e.g.
`openai/gpt-oss-120b`) support JSON mode but not parallel tool calls, and a
single JSON object per chunk is a pure function of (model, chunk) modulo the
network call itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import SystemMessage, HumanMessage

from ingestion.extraction.prompts import (
    CHUNK_TASK_TEMPLATE,
    SYSTEM_PROMPT,
)
from ingestion.extraction.schema import ExtractionDocument
from ingestion.extraction.tools import document_to_extraction
from ingestion.pipeline_types import ChunkExtraction


class JsonModeChatModel(Protocol):
    """Structural type for whatever chat model the app configures (see
    supervisor_plan.md's 'select the LLM provider and model' requirement)."""

    def bind(self, **kwargs) -> "JsonModeChatModel": ...

    def invoke(self, messages: list) -> object: ...  # returns an AIMessage


@dataclass(frozen=True, slots=True)
class ExtractionAgent:
    """A bound, ready-to-invoke model. Immutable -- build once, reuse."""

    model: JsonModeChatModel


def build_extraction_agent(llm: JsonModeChatModel) -> ExtractionAgent:
    return ExtractionAgent(
        model=llm.bind(response_format={"type": "json_object"})
    )


def _build_messages(*, source_name: str, chunk_index: int, chunk_text: str) -> list:
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=CHUNK_TASK_TEMPLATE.format(
                source_name=source_name,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
            )
        ),
    ]


def _parse_extraction(chunk_index: int, raw: str) -> ChunkExtraction:
    """Parse the model's JSON into a ChunkExtraction. Malformed output maps
    to an empty extraction rather than raising, so extraction stays total."""
    try:
        doc = ExtractionDocument.model_validate_json(raw)
    except Exception:
        return ChunkExtraction(chunk_index=chunk_index, entity=None)
    return document_to_extraction(chunk_index, doc)


def extract_chunk(
    agent: ExtractionAgent,
    *,
    source_name: str,
    chunk_index: int,
    chunk_text: str,
) -> ChunkExtraction:
    """
    Run extraction for a single chunk. The only I/O is the one model call;
    everything before and after it is pure (see prompts.py / tools.py).
    """
    messages = _build_messages(
        source_name=source_name, chunk_index=chunk_index, chunk_text=chunk_text
    )
    response = agent.model.invoke(messages)
    return _parse_extraction(chunk_index, getattr(response, "content", "") or "")


def extract_document(
    agent: ExtractionAgent,
    *,
    source_name: str,
    chunks: tuple,  # tuple[chunk_embed.types.EmbeddedChunk, ...]
) -> tuple[ChunkExtraction, ...]:
    """
    Run extraction across every chunk of a document. Reads the real
    EmbeddedChunk shape (`embedded.chunk.chunk_index` / `.text`) directly --
    no adapter object needed. A comprehension, not a for-loop with an
    accumulator list, since each chunk's extraction is independent (step 5
    is scoped per-chunk).
    """
    return tuple(
        extract_chunk(
            agent,
            source_name=source_name,
            chunk_index=embedded.chunk.chunk_index,
            chunk_text=embedded.chunk.text,
        )
        for embedded in chunks
    )
