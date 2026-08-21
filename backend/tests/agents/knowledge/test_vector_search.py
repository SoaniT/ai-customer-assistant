"""Tests for the relaxed second-pass retrieval in vector_search.

Covers the recall recovery added for count/aggregate-style queries
("how many members are there in alpinist" scores ~0.66 against the team
chunk, just under the 0.7 primary threshold): when the primary pass
returns nothing, a second pass at `relaxed_similarity_threshold` runs;
only if even that yields nothing does retrieval raise EmptyRetrievalError.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.knowledge.config import KnowledgeAgentConfig
from agents.knowledge.constants import (
    DEFAULT_RELAXED_SIMILARITY_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from agents.knowledge.exceptions import EmptyRetrievalError
from agents.knowledge.types import RewrittenQuery
from agents.knowledge.vector_search import vector_search


def _row(embedding: tuple[float, ...], *, text: str = "some text") -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id=f"chunk-{id(embedding)}",
        text=text,
        embedding=embedding,
        chunk_index=0,
        page=1,
        source_name="about",
        source_type="web",
        category_name=None,
        version_number=2,
        entity_type=None,
        entity_label=None,
    )


def _embed(query_vector: tuple[float, ...]):
    def embed_query(_text: str) -> tuple[float, ...]:
        return query_vector
    return embed_query


@pytest.fixture
def config() -> KnowledgeAgentConfig:
    return KnowledgeAgentConfig(embedding_dimension=2)


class TestRelaxedSecondPass:
    def test_recovers_chunk_below_primary_but_at_or_above_relaxed(self, config, monkeypatch):
        # Query [1, 0]; row A cosine 0.6 (primary miss, relaxed hit),
        # row B cosine 0.4 (miss on both passes).
        rows = [
            _row((0.4, 0.9), text="b-miss"),
            _row((0.6, 0.8), text="a-relaxed-hit"),
        ]
        async def fake_fetch_rows(_session, _statement):
            return rows
        monkeypatch.setattr(
            "agents.knowledge.vector_search._fetch_rows", fake_fetch_rows
        )

        rewritten = RewrittenQuery(
            original_text="how many members are there in alpinist",
            rewritten_text="how many members are there in alpinist",
        )
        result = _await(
            vector_search(
                rewritten,
                config=config,
                session=None,
                embed_query=_embed((1.0, 0.0)),
            )
        )

        assert len(result) == 1
        assert result[0].chunk_text == "a-relaxed-hit"
        assert result[0].similarity_score == pytest.approx(0.6, abs=0.0001)

    def test_primary_pass_never_runs_relaxed(self, config, monkeypatch):
        rows = [_row((0.8, 0.6), text="primary-hit")]
        async def fake_fetch_rows(_session, _statement):
            return rows
        monkeypatch.setattr(
            "agents.knowledge.vector_search._fetch_rows", fake_fetch_rows
        )

        rewritten = RewrittenQuery(
            original_text="who is the ceo of alpinist",
            rewritten_text="who is the ceo of alpinist",
        )
        result = _await(
            vector_search(
                rewritten,
                config=config,
                session=None,
                embed_query=_embed((1.0, 0.0)),
            )
        )

        assert len(result) == 1
        assert result[0].chunk_text == "primary-hit"
        assert result[0].similarity_score == pytest.approx(0.8, abs=0.0001)

    def test_still_raises_when_even_relaxed_threshold_is_met_by_nothing(self, config, monkeypatch):
        rows = [_row((0.4, 0.9), text="way-off")]
        async def fake_fetch_rows(_session, _statement):
            return rows
        monkeypatch.setattr(
            "agents.knowledge.vector_search._fetch_rows", fake_fetch_rows
        )

        rewritten = RewrittenQuery(
            original_text="what is the meaning of life",
            rewritten_text="what is the meaning of life",
        )
        with pytest.raises(EmptyRetrievalError):
            _await(
                vector_search(
                    rewritten,
                    config=config,
                    session=None,
                    embed_query=_embed((1.0, 0.0)),
                )
            )


class TestConfigDefaults:
    def test_defaults_keep_primary_strict_and_relaxed_below(self):
        config = KnowledgeAgentConfig()
        assert config.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
        assert config.relaxed_similarity_threshold == DEFAULT_RELAXED_SIMILARITY_THRESHOLD
        assert config.relaxed_similarity_threshold < config.similarity_threshold

    def test_rejects_relaxed_threshold_above_primary(self):
        with pytest.raises(ValueError):
            KnowledgeAgentConfig(
                similarity_threshold=0.6,
                relaxed_similarity_threshold=0.7,
            )


def _await(coro):
    import asyncio
    return asyncio.run(coro)
