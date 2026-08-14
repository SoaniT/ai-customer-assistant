"""HTTP ingestion endpoints.

Both paths reuse the exact orchestration the CLI worker uses
(``scripts/crawl_and_ingest.py``): register a document version (checksum
dedup, MinIO storage, queue row) then run ``run_ingestion`` immediately so
the result is indexable without a separate worker process.

  POST /ingest/upload   multipart file (PDF / DOCX / Markdown)
  POST /ingest/crawl    JSON {"url": ..., "site": false}
                        (HTML page or PDF/Office URL; ``site=true`` crawls
                        the whole domain, one job per page)

Uploaded-by defaults to the system service account
(``00000000-0000-0000-0000-000000000000``); override with
``INGEST_DEFAULT_USER_ID`` in the environment.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from db.async_session import get_session, session_factory
from ingestion.pipeline_types import FileType, JobType, JobRef, JobStatus
from ingestion.queue.document_producer import register_document_version

router = APIRouter(prefix="/ingest", tags=["ingest"])

DEFAULT_USER_ID = UUID("00000000-0000-0000-0000-000000000000")


def _uploaded_by() -> UUID:
    raw = os.environ.get("INGEST_DEFAULT_USER_ID")
    return UUID(raw) if raw else DEFAULT_USER_ID


_UPLOAD_MIME_TO_FILE_TYPE: dict[str, FileType] = {
    "application/pdf": FileType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCX,
    "text/markdown": FileType.MD,
}


async def _run_job(job_id: UUID) -> None:
    """Execute a queued ingestion job synchronously (mirrors the CLI worker)."""
    from db.models import KnowledgeInjectionJob
    from ingestion.pipeline import run_ingestion
    from ingestion.queue import repository as job_repo

    async with session_factory() as session:
        row = await session.get(KnowledgeInjectionJob, job_id)
        if row is None:
            return
        job = JobRef(
            job_id=row.job_id,
            source_id=row.source_id,
            version_id=row.version_id,
            job_type=JobType(row.job_type),
            status=JobStatus(row.status),
            triggered_by=row.triggered_by,
        )
        try:
            outcome = await run_ingestion(session, job)
        except Exception as exc:  # noqa: BLE001
            await job_repo.complete_job(
                session,
                job_id=job.job_id,
                status=JobStatus.FAILED,
                chunks_created_count=0,
                entities_created_count=0,
                error_details=f"unhandled_exception: {exc}",
            )
        else:
            await job_repo.complete_job(
                session,
                job_id=outcome.job_id,
                status=outcome.status,
                chunks_created_count=outcome.chunks_created_count,
                entities_created_count=outcome.entities_created_count,
                error_details=outcome.error_details,
            )
        await session.commit()


async def _register_and_run(
    session: AsyncSession,
    *,
    url: str,
    raw_bytes: bytes,
    mime_type: str,
    file_type: FileType | None,
    category_id: UUID | None,
) -> dict:
    job = await register_document_version(
        session,
        url=url,
        raw_bytes=raw_bytes,
        mime_type=mime_type,
        file_type=file_type,
        uploaded_by=_uploaded_by(),
        category_id=category_id,
    )
    if job is None:
        return {"status": "duplicate_skipped"}
    asyncio.create_task(_run_job(job.job_id))
    return {
        "status": "submitted",
        "job_id": str(job.job_id),
        "source_id": str(job.source_id),
        "version_id": str(job.version_id),
    }


@router.get("/jobs/{job_id}")
async def job_status(job_id: UUID, session: AsyncSession = Depends(get_session)) -> dict:
    from db.models import KnowledgeInjectionJob

    row = await session.get(KnowledgeInjectionJob, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": str(row.job_id),
        "status": row.status,
        "chunks_created_count": row.chunks_created_count,
        "entities_created_count": row.entities_created_count,
        "error_details": row.error_details,
    }


@router.post("/upload")
async def upload(
    file: UploadFile,
    category_id: UUID | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict:
    mime = (file.content_type or "application/octet-stream").split(";", 1)[0].strip().lower()
    file_type = _UPLOAD_MIME_TO_FILE_TYPE.get(mime)
    if file_type is None:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{mime}'. Expected PDF, DOCX or Markdown.",
        )
    name = (file.filename or "upload").rsplit("/", 1)[-1]
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    return await _register_and_run(
        session,
        url=f"manual_upload://{name}",
        raw_bytes=data,
        mime_type=mime,
        file_type=file_type,
        category_id=category_id,
    )


class CrawlRequest(BaseModel):
    url: str = Field(..., min_length=5, max_length=2048)
    category_id: UUID | None = None
    site: bool = Field(
        default=False,
        description="Crawl the entire site under the URL's domain instead of just this page.",
    )


@router.post("/crawl")
async def crawl(
    req: CrawlRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    from ingestion.services.crawl_ingest import crawl_and_register_jobs

    try:
        result = await crawl_and_register_jobs(
            session,
            url=req.url,
            uploaded_by=_uploaded_by(),
            category_id=req.category_id,
            site=req.site,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}") from exc

    jobs = result.jobs
    if jobs:
        for job in jobs:
            asyncio.create_task(_run_job(job.job_id))

        if len(jobs) == 1:
            job = jobs[0]
            resp: dict = {
                "status": "submitted",
                "job_id": str(job.job_id),
                "source_id": str(job.source_id),
                "version_id": str(job.version_id),
            }
        else:
            resp = {
                "status": "submitted",
                "job_ids": [str(job.job_id) for job in jobs],
                "source_ids": [str(job.source_id) for job in jobs],
                "version_ids": [str(job.version_id) for job in jobs],
                "pages": len(jobs),
            }
        if result.failures:
            resp["failed_pages"] = len(result.failures)
        return resp

    if result.failures:
        failed_url, reason = result.failures[0]
        detail = f"Crawl failed for {failed_url}: {reason}"
        if len(result.failures) > 1:
            detail += f" ({len(result.failures) - 1} more page(s) failed)"
        raise HTTPException(status_code=502, detail=detail)

    return {"status": "duplicate_skipped"}