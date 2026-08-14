"""Unit tests for the shared crawl-to-jobs orchestration.

The routing decision (HTML page vs whole-site crawl vs raw document) lives
in ingestion/services/crawl_ingest.py, shared by the CLI and the HTTP API.
These tests exercise the routing and registration logic with the crawler
and the DB-touching ``register_document_version`` stubbed out -- no network,
no Postgres.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from ingestion.crawler.config import CrawlMode
from ingestion.crawler.models import CrawlDocument
from ingestion.pipeline_types import FileType
from ingestion.services import crawl_ingest as ci


class _FakeCrawler:
    docs: list[CrawlDocument] = []
    url: str | None = None
    last_config = None

    def __init__(self, config):
        self.config = config
        type(self).last_config = config

    async def crawl(self, url: str) -> list[CrawlDocument]:
        type(self).url = url
        return list(type(self).docs)


class _FakeJob:
    def __init__(self, job_id):
        self.job_id = job_id


def _doc(url: str = "https://example.com/", markdown: str = "# hi", error: str | None = None) -> CrawlDocument:
    return CrawlDocument(
        url=url, title="", markdown=markdown, html="", depth=0, status_code=200, error=error
    )


def _patch_fetch(monkeypatch, url: str, content_type: str, raw: bytes = b"<html></html>") -> None:
    async def fake_fetch(url_: str) -> ci.FetchedUrl:
        return ci.FetchedUrl(url=url_, content_type=content_type, raw_bytes=raw)

    monkeypatch.setattr(ci, "fetch_url", fake_fetch)


def _patch_register(monkeypatch, *, always_duplicate: bool = False):
    calls = []

    async def fake_register(
        session, *, url, raw_bytes, mime_type, file_type, uploaded_by, category_id
    ):
        calls.append((url, raw_bytes, mime_type, file_type))
        return None if always_duplicate else _FakeJob(job_id=uuid4())

    monkeypatch.setattr(ci, "register_document_version", fake_register)
    return calls


def test_classify_content_type_html_vs_document():
    assert ci.classify_content_type("text/html") == "html"
    assert ci.classify_content_type("text/html; charset=utf-8") == "html"
    assert ci.classify_content_type("application/pdf") == "document"
    assert (
        ci.classify_content_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        == "document"
    )


def test_resolve_file_type_maps_documents():
    assert ci.resolve_file_type("application/pdf") is FileType.PDF
    assert ci.resolve_file_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document") is FileType.DOCX
    assert ci.resolve_file_type("text/html") is None


@pytest.mark.asyncio
async def test_document_url_registers_raw_bytes(monkeypatch):
    _patch_fetch(monkeypatch, "https://example.com/a.pdf", "application/pdf", raw=b"%PDF-1.4")
    calls = _patch_register(monkeypatch)

    jobs = (await ci.crawl_and_register_jobs(
        None, url="https://example.com/a.pdf", uploaded_by=uuid4(), category_id=None
    )).jobs

    assert len(jobs) == 1
    url, raw, mime, file_type = calls[0]
    assert url == "https://example.com/a.pdf"
    assert raw == b"%PDF-1.4"
    assert mime == "application/pdf"
    assert file_type is FileType.PDF


@pytest.mark.asyncio
async def test_page_mode_crawls_one_page_and_registers_markdown(monkeypatch):
    _patch_fetch(monkeypatch, "https://example.com/", "text/html")
    _FakeCrawler.docs = [_doc(url="https://example.com/", markdown="# Home page")]
    monkeypatch.setattr(ci, "Crawler", _FakeCrawler)
    calls = _patch_register(monkeypatch)

    result = await ci.crawl_and_register_jobs(
        None, url="https://example.com/", uploaded_by=uuid4(), category_id=None, site=False
    )

    assert len(result.jobs) == 1
    assert _FakeCrawler.last_config.mode is CrawlMode.PAGE
    assert _FakeCrawler.last_config.allowed_domains == ("example.com",)
    url, raw, mime, file_type = calls[0]
    assert url == "https://example.com/"
    assert raw == b"# Home page"
    assert mime == "text/markdown"
    assert file_type is FileType.MD


@pytest.mark.asyncio
async def test_site_mode_uses_site_crawl(monkeypatch):
    _patch_fetch(monkeypatch, "https://example.com/", "text/html")
    _FakeCrawler.docs = [
        _doc(url="https://example.com/", markdown="# Home"),
        _doc(url="https://example.com/about", markdown="# About"),
    ]
    monkeypatch.setattr(ci, "Crawler", _FakeCrawler)
    calls = _patch_register(monkeypatch)

    result = await ci.crawl_and_register_jobs(
        None, url="https://example.com/", uploaded_by=uuid4(), category_id=None, site=True
    )

    assert _FakeCrawler.last_config.mode is CrawlMode.SITE
    assert len(result.jobs) == 2
    assert result.failures == ()
    assert [c[0] for c in calls] == ["https://example.com/", "https://example.com/about"]


@pytest.mark.asyncio
async def test_failed_pages_are_skipped_not_registered(monkeypatch):
    _patch_fetch(monkeypatch, "https://example.com/", "text/html")
    _FakeCrawler.docs = [_doc(error="boom"), _doc(url="https://example.com/ok", markdown="# ok")]
    monkeypatch.setattr(ci, "Crawler", _FakeCrawler)
    calls = _patch_register(monkeypatch)

    result = await ci.crawl_and_register_jobs(
        None, url="https://example.com/", uploaded_by=uuid4(), category_id=None, site=True
    )

    assert len(result.jobs) == 1
    assert result.failures == (("https://example.com/", "boom"),)
    assert [c[0] for c in calls] == ["https://example.com/ok"]


@pytest.mark.asyncio
async def test_all_pages_failed_reports_failures_not_duplicates(monkeypatch):
    _patch_fetch(monkeypatch, "https://example.com/", "text/html")
    _FakeCrawler.docs = [
        _doc(
            url="https://example.com/",
            error="Trafilatura could not extract content from https://example.com/",
        )
    ]
    monkeypatch.setattr(ci, "Crawler", _FakeCrawler)
    calls = _patch_register(monkeypatch)

    result = await ci.crawl_and_register_jobs(
        None, url="https://example.com/", uploaded_by=uuid4(), category_id=None
    )

    assert result.jobs == ()
    assert result.failures == (
        ("https://example.com/", "Trafilatura could not extract content from https://example.com/"),
    )
    assert calls == []


@pytest.mark.asyncio
async def test_crawler_returning_no_documents_is_a_failure(monkeypatch):
    _patch_fetch(monkeypatch, "https://example.com/", "text/html")
    _FakeCrawler.docs = []
    monkeypatch.setattr(ci, "Crawler", _FakeCrawler)
    _patch_register(monkeypatch)

    result = await ci.crawl_and_register_jobs(
        None, url="https://example.com/", uploaded_by=uuid4(), category_id=None
    )

    assert result.jobs == ()
    assert result.failures == (("https://example.com/", "crawler returned no documents"),)


@pytest.mark.asyncio
async def test_all_duplicates_returns_empty(monkeypatch):
    _patch_fetch(monkeypatch, "https://example.com/", "text/html")
    _FakeCrawler.docs = [_doc(markdown="# dup")]
    monkeypatch.setattr(ci, "Crawler", _FakeCrawler)
    _patch_register(monkeypatch, always_duplicate=True)

    result = await ci.crawl_and_register_jobs(
        None, url="https://example.com/", uploaded_by=uuid4(), category_id=None
    )

    assert result.jobs == ()
    assert result.failures == ()