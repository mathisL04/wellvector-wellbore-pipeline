"""Async PDF downloader with caching and parallel execution."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import httpx
from rich.progress import Progress, TaskID

from pipeline.models import DocumentMeta


DEFAULT_CACHE_DIR = Path("data/pdfs")
MAX_CONCURRENT_DOWNLOADS = 10
DOWNLOAD_TIMEOUT = 60.0


def _cache_path(doc: DocumentMeta, cache_dir: Path) -> Path:
    """Deterministic local path for a downloaded PDF."""
    safe_name = doc.doc_name.replace("/", "_").replace(" ", "_")
    return cache_dir / f"{safe_name}.pdf"


async def _download_one(
    client: httpx.AsyncClient,
    doc: DocumentMeta,
    cache_dir: Path,
    semaphore: asyncio.Semaphore,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> tuple[DocumentMeta, Path | None]:
    """Download a single PDF, skip if cached. Returns (doc, local_path)."""
    dest = _cache_path(doc, cache_dir)

    if dest.exists() and dest.stat().st_size > 0:
        if progress and task_id is not None:
            progress.advance(task_id)
        return doc, dest

    async with semaphore:
        try:
            resp = await client.get(doc.url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            if progress and task_id is not None:
                progress.advance(task_id)
            return doc, dest
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            if progress and task_id is not None:
                progress.advance(task_id)
            return doc, None


async def download_all(
    documents: list[DocumentMeta],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_concurrent: int = MAX_CONCURRENT_DOWNLOADS,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> list[tuple[DocumentMeta, Path | None]]:
    """Download all PDFs in parallel with concurrency limit.

    Returns list of (document_metadata, local_path_or_None).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)

    async with httpx.AsyncClient(
        headers={"User-Agent": "WellExtractionPipeline/1.0"},
        follow_redirects=True,
    ) as client:
        tasks = [
            _download_one(client, doc, cache_dir, semaphore, progress, task_id)
            for doc in documents
        ]
        return await asyncio.gather(*tasks)
