#!/usr/bin/env python
"""
Crawl a URL and ingest it end-to-end.

    uv run --project backend python scripts/crawl_and_ingest.py https://example.com/handbook
    uv run --project backend python scripts/crawl_and_ingest.py --site https://example.com
    uv run --project backend python scripts/crawl_and_ingest.py https://example.com/handbook.pdf

Routing is decided purely by the fetched Content-Type (a pure function,
`classify_content_type` in ingestion/services/crawl_ingest.py):

  * text/html                          -> ingestion.crawler.Crawler (PAGE or
                                           SITE mode) converts the page(s) to
                                           markdown via trafilatura; each page
                                           is registered as an MD version and
                                           flows through the SAME pipeline as
                                           everything else.
  * pdf / doc / docx / ppt / pptx /
    xls / xlsx (any Office or PDF type) -> raw bytes are registered as-is;
                                           Tika extracts the text during
                                           ingestion, same as an uploaded file.

Each registered version produces a queued `knowledge_injection_job`, which
this script then runs immediately through the same `run_ingestion` the
background worker uses -- so a one-shot CLI call gives you fully indexed
document(s) without needing the worker running. A site crawl (--site)
registers one job per crawled page, capped at CrawlConfig.max_pages.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import async_sessionmaker

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # backend/.env

# Add src/ai_customer_assistant to sys.path so bare `ingestion.xxx` / `db.xxx`
# imports work when this script is invoked directly (matches pyproject.toml's
# pytest pythonpath convention, and how alembic/env.py resolves `db.models`).
_PKG_ROOT = Path(__file__).resolve().parent.parent / "src" / "ai_customer_assistant"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ingestion.pipeline_types import JobRef, JobStatus, JobType
from ingestion.services.crawl_ingest import crawl_and_register_jobs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def crawl_and_ingest(
    url: str,
    *,
    uploaded_by: UUID,
    category_id: UUID | None,
    session_factory: async_sessionmaker,
    site: bool = False,
) -> None:
    async with session_factory() as session:
        result = await crawl_and_register_jobs(
            session,
            url=url,
            uploaded_by=uploaded_by,
            category_id=category_id,
            site=site,
        )
        job_rows = tuple(result.jobs)
        failures = tuple(result.failures)

    for failed_url, reason in failures:
        logger.error("crawl failed for %s: %s", failed_url, reason)

    if not job_rows:
        return

    logger.info("%d page(s) queued from %s -- ingesting each now", len(job_rows), url)

    from ingestion.pipeline import run_ingestion
    from ingestion.queue import repository as job_repo

    for row in job_rows:
        job = JobRef(
            job_id=row.job_id,
            source_id=row.source_id,
            version_id=row.version_id,
            job_type=JobType(row.job_type),
            status=JobStatus(row.status),
            triggered_by=row.triggered_by,
        )
        async with session_factory() as session:
            try:
                outcome = await run_ingestion(session, job)
            except Exception as exc:  # noqa: BLE001
                logger.exception("ingestion crashed for job %s", job.job_id)
                await job_repo.complete_job(
                    session,
                    job_id=job.job_id,
                    status=JobStatus.FAILED,
                    chunks_created_count=0,
                    entities_created_count=0,
                    error_details=f"unhandled_exception: {exc}",
                )
                continue
            logger.info(
                "ingestion %s for job %s: chunks=%d entities=%d %s",
                outcome.status.value, job.job_id, outcome.chunks_created_count,
                outcome.entities_created_count, outcome.error_details or "",
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", action="store_true", help="crawl the entire site (CrawlMode.SITE), not just this one page",)
    parser.add_argument("url", help="URL to crawl and ingest")
    parser.add_argument("--uploaded-by", required=True, help="app_user.id (a UUID) of the service account")
    parser.add_argument("--category-id", default=None, help="knowledge_category.category_id, optional")
    return parser.parse_args()


def _async_database_url() -> str:
    from db.session import database_url

    sync_url = database_url()
    return sync_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Quiet third-party libraries' own INFO logging (huggingface_hub's model
    # cache checks, httpx's per-request logging for every HF/Tika/Groq call)
    # -- keeps the log to your app's own ingestion progress lines.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    args = _parse_args()

    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(_async_database_url())
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    asyncio.run(
        crawl_and_ingest(
            args.url,
            uploaded_by=UUID(args.uploaded_by),
            category_id=UUID(args.category_id) if args.category_id else None,
            session_factory=session_factory,
            site=args.site,
        )
    )


if __name__ == "__main__":
    main()
