"""Background worker that processes extraction jobs from the queue."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from typing import Optional

from app.config import WORKERS, JOB_TIMEOUT_SECONDS
from app.database import JobStore
from app.llm.client import UsageStats
from app.models import (
    ExtractionMetadata,
    ExtractionSchema,
    FieldResult,
    JobStatus,
    SchemaField,
)
from app.pipeline.extractor import extract_document

logger = logging.getLogger(__name__)


class WorkerPool:
    """Manages a pool of async workers that process extraction jobs."""

    def __init__(self, store: JobStore, max_workers: int = WORKERS):
        self.store = store
        self.max_workers = max_workers
        self._queue: asyncio.Queue = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self):
        """Start the worker pool."""
        self._running = True
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(i))
            self._tasks.append(task)
        # Also start the recovery task to pick up any pending jobs from DB
        asyncio.create_task(self._recover_pending())
        logger.info(f"Worker pool started with {self.max_workers} workers")

    async def stop(self):
        """Stop the worker pool gracefully."""
        self._running = False
        # Signal workers to stop
        for _ in self._tasks:
            await self._queue.put(None)
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Worker pool stopped")

    async def submit(self, job_id: str):
        """Submit a job for processing."""
        await self._queue.put(job_id)
        logger.info(f"Job {job_id} submitted to worker queue")

    async def _recover_pending(self):
        """Recover any PENDING jobs from the database (after a restart)."""
        try:
            pending = await self.store.get_pending_jobs()
            for job in pending:
                logger.info(f"Recovering pending job: {job['job_id']}")
                await self._queue.put(job["job_id"])
        except Exception as e:
            logger.error(f"Failed to recover pending jobs: {e}")

    async def _worker(self, worker_id: int):
        """Single worker loop: pull jobs from queue and process them."""
        logger.info(f"Worker {worker_id} started")
        while self._running:
            try:
                job_id = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            if job_id is None:
                break  # Shutdown signal

            try:
                await self._process_job(job_id, worker_id)
            except Exception as e:
                logger.error(f"Worker {worker_id} failed on job {job_id}: {e}\n{traceback.format_exc()}")
                try:
                    await self.store.update_status(
                        job_id,
                        JobStatus.FAILED,
                        error={"code": "PROCESSING_ERROR", "message": str(e)},
                    )
                except Exception:
                    pass
            finally:
                self._queue.task_done()

        logger.info(f"Worker {worker_id} stopped")

    async def _process_job(self, job_id: str, worker_id: int):
        """Process a single extraction job."""
        start_time = time.time()

        job = await self.store.get_job(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found in database")
            return

        # Skip if already terminal
        if job["status"] in (JobStatus.SUCCESS, JobStatus.PARTIAL, JobStatus.FAILED):
            logger.info(f"Job {job_id} already terminal ({job['status']}), skipping")
            return

        logger.info(f"Worker {worker_id} processing job {job_id}: {job['filename']}")

        # Update to RUNNING
        await self.store.update_status(job_id, JobStatus.RUNNING, stage="STARTING")

        # Parse schema
        schema_data = json.loads(job["schema_json"])
        schema = ExtractionSchema(
            name=schema_data.get("name", "unknown"),
            description=schema_data.get("description", ""),
            fields={
                k: SchemaField(**v) for k, v in schema_data.get("fields", {}).items()
            },
        )

        # Track usage
        usage = UsageStats()

        # Stage update callback
        async def update_stage(stage: str):
            await self.store.update_status(job_id, JobStatus.RUNNING, stage=stage)

        try:
            # Run extraction with timeout
            entries = await asyncio.wait_for(
                extract_document(
                    file_path=job["file_path"],
                    filename=job["filename"],
                    schema=schema,
                    usage_stats=usage,
                    update_stage=update_stage,
                ),
                timeout=JOB_TIMEOUT_SECONDS,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Build result
            result = {
                "schema": schema.name,
                "entries": [
                    {k: v.model_dump(exclude_none=True) for k, v in entry.items()}
                    for entry in entries
                ],
                "metadata": {
                    **usage.to_metadata(),
                    "latency_ms": elapsed_ms,
                },
            }

            # Check if we have any entries with all nulls → PARTIAL
            has_data = any(
                any(v.value is not None for v in entry.values())
                for entry in entries
            )
            status = JobStatus.SUCCESS if has_data else JobStatus.PARTIAL

            await self.store.update_status(job_id, status, result=result)
            logger.info(
                f"Job {job_id} completed: status={status}, entries={len(entries)}, "
                f"cost=${usage.cost_usd:.4f}, latency={elapsed_ms}ms"
            )

        except asyncio.TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            await self.store.update_status(
                job_id,
                JobStatus.FAILED,
                error={"code": "TIMEOUT", "message": f"Job timed out after {JOB_TIMEOUT_SECONDS}s"},
            )
            logger.error(f"Job {job_id} timed out after {elapsed_ms}ms")

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Job {job_id} failed: {e}\n{traceback.format_exc()}")
            await self.store.update_status(
                job_id,
                JobStatus.FAILED,
                error={"code": "EXTRACTION_ERROR", "message": str(e)},
            )

