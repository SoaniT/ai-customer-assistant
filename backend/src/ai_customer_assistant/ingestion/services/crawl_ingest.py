"""
Crawl a URL (optionally the whole site) and register document version(s).

This module is the single place that decides how a fetched URL is turned
into queued ingestion jobs, shared by the CLI (scripts/crawl_and_ingest.py)
and the HTTP API (api/ingest.py):

  * text/html -> ingestion.crawler.Crawler in PAGE or SITE mode; the page(s)
                 are converted to markdown and each one is registered as an
                 MD version.
  * PDF / Office document -> raw bytes are registered as-is; Tika extracts
                 the text during ingestion.

The returned jobs are queued but NOT executed here -- callers run them
with run_ingestion (CLI) or a background task (API).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from ingestion.crawler.config import CrawlConfig, CrawlMode
from ingestion.crawler.crawler import Crawler
from ingestion.crawler.models import CrawlDocument
from ingestion.pipeline_types import FileType
from ingestion.queue.document_producer import register_document_version

logger = logging.getLogger(__name__)

RouteKind = Literal["html", "document"]

_DOCUMENT_MIME_TO_FILE_TYPE: dict[str, FileType | None] = {
    "application/pdf": FileType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCX,
    "application/msword": None,  # legacy .doc -- no enum value
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": None,  # .pptx
    "application/vnd.ms-powerpoint": None,  # .ppt
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": None,  # .xlsx
    "application/vnd.ms-excel": None,  # .xls
}


def classify_content_type(content_type: str) -> RouteKind:
    """Pure: decide which ingestion path a fetched Content-Type takes."""
    base_type = content_type.split(";", 1)[0].strip().lower()
    return "document" if base_type in _DOCUMENT_MIME_TO_FILE_TYPE else "html"


def resolve_file_type(content_type: str) -> FileType | None:
    """Pure: map a document mime type to the schema's (limited) enum."""
    base_type = content_type.split(";", 1)[0].strip().lower()
    return _DOCUMENT_MIME_TO_FILE_TYPE.get(base_type)


@dataclass(frozen=True, slots=True)
class FetchedUrl:
    url: str
    content_type: str
    raw_bytes: bytes


@dataclass(frozen=True, slots=True)
class CrawlResult:
    """Outcome of a crawl-and-register attempt.

    ``jobs`` holds the queued ``KnowledgeInjectionJob`` rows (empty when every
    page was duplicate content *or* nothing could be registered). ``failures``
    lists ``(url, reason)`` for pages whose content could not be fetched or
    extracted -- callers must surface these instead of assuming an empty job
    list means "everything was a duplicate".
    """

    jobs: tuple[object, ...] = ()
    failures: tuple[tuple[str, str], ...] = ()


async def fetch_url(url: str) -> FetchedUrl:
    """Sniff the URL's Content-Type to decide routing (HTML vs document).

    Kept as a plain httpx call rather than the crawler's own fetcher: this
    only decides whether to hand off to Crawler at all. ``url`` is the
    post-redirect final URL, used as the source identity when registering.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "application/octet-stream")
        return FetchedUrl(url=str(response.url), content_type=content_type, raw_bytes=response.content)


async def crawl_and_register_jobs(
    session: AsyncSession,
    *,
    url: str,
    uploaded_by: UUID,
    category_id: UUID | None,
    site: bool = False,
) -> CrawlResult:
    """Fetch the URL and register one queued job per document to ingest.

    ``site=True`` crawls the whole domain (CrawlMode.SITE) instead of just
    the one page (CrawlMode.PAGE). Returns a :class:`CrawlResult` carrying
    the queued ``KnowledgeInjectionJob`` ORM rows plus the URLs whose
    content could not be extracted (with the reason). Fetches already-visited
    URLs exactly once thanks to the crawler's own frontier bookkeeping.
    """
    fetched = await fetch_url(url)
    route = classify_content_type(fetched.content_type)

    if route == "html":
        return await _register_html(session, fetched, uploaded_by=uploaded_by, category_id=category_id, site=site)

    job = await register_document_version(
        session,
        url=fetched.url,
        raw_bytes=fetched.raw_bytes,
        mime_type=fetched.content_type,
        file_type=resolve_file_type(fetched.content_type),
        uploaded_by=uploaded_by,
        category_id=category_id,
    )
    if job is None:
        return CrawlResult()
    return CrawlResult(jobs=(job,))


async def _register_html(
    session: AsyncSession,
    fetched: FetchedUrl,
    *,
    uploaded_by: UUID,
    category_id: UUID | None,
    site: bool,
) -> CrawlResult:
    domain = urlsplit(fetched.url).netloc
    config = CrawlConfig(mode=CrawlMode.SITE if site else CrawlMode.PAGE, allowed_domains=(domain,))
    documents: tuple[CrawlDocument, ...] = await Crawler(config).crawl(fetched.url)

    jobs: list[object] = []
    failures: list[tuple[str, str]] = []
    for doc in documents:
        if doc.error is not None:
            failures.append((doc.url, doc.error))
            logger.error("crawl failed for %s: %s", doc.url, doc.error)
            continue
        job = await register_document_version(
            session,
            url=doc.url,
            raw_bytes=doc.markdown.encode("utf-8"),
            mime_type="text/markdown",
            file_type=FileType.MD,
            uploaded_by=uploaded_by,
            category_id=category_id,
        )
        if job is not None:
            jobs.append(job)

    if not documents:
        failures.append((fetched.url, "crawler returned no documents"))

    return CrawlResult(jobs=tuple(jobs), failures=tuple(failures))