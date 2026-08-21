"""make embedding_chunk writes idempotent

A single ingestion job could be executed twice (the API's in-process runner
and the queue worker both consumed the same QUEUED job -- see
ingestion/queue/repository.claim_job). persist_chunks had no unique
constraint and did plain inserts, so the second run inserted duplicate
rows for the same (version_id, chunk_index), and _link_entity_to_chunk's
.scalar_one() then failed with MultipleResultsFound ("persist_failed").

This migration:
  1. collapses existing duplicate chunk rows (keeps the lowest chunk_id per
     (version_id, chunk_index)),
  2. adds the unique constraint matching the new ON CONFLICT DO NOTHING
     writes in ingestion/persistence.py.

Run after b4e2d9a1c7f3. Idempotent -- a second run finds nothing to merge.

Revision ID: c7d5a3b2e1f0
Revises: b4e2d9a1c7f3
Create Date: 2026-08-17 01:00:00.000000
"""
from typing import Sequence, Union

from alembic import op

revision: str = "c7d5a3b2e1f0"
down_revision: Union[str, None] = "b4e2d9a1c7f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Keep one row per (version_id, chunk_index) -- the lowest chunk_id is
    # the first one written.
    op.execute(
        """
        DELETE FROM embedding_chunk a USING embedding_chunk b
        WHERE a.chunk_id > b.chunk_id
          AND a.version_id = b.version_id
          AND a.chunk_index = b.chunk_index
        """
    )
    op.create_unique_constraint(
        "uq_embedding_chunk_version_index",
        "embedding_chunk",
        ["version_id", "chunk_index"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_embedding_chunk_version_index", "embedding_chunk", type_="unique")