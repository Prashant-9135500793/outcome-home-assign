"""SQLite-backed job store with async interface."""
from __future__ import annotations

import json
import os

import aiosqlite
from datetime import datetime, timezone
from typing import Optional

from app.config import DB_PATH
from app.models import JobStatus


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    status       TEXT NOT NULL DEFAULT 'PENDING',
    stage        TEXT,
    org_id       TEXT,
    file_hash    TEXT,
    schema_hash  TEXT,
    schema_name  TEXT,
    filename     TEXT,
    file_path    TEXT,
    schema_json  TEXT,
    result       TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_file_schema ON jobs(file_hash, schema_hash, org_id);
"""


class JobStore:
    """Thin async wrapper around SQLite for job persistence."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_CREATE_TABLE)
        await self._db.commit()
        # Recover any jobs that were RUNNING when we crashed
        await self._db.execute(
            "UPDATE jobs SET status = 'PENDING', stage = NULL, updated_at = ? "
            "WHERE status IN ('PENDING', 'RUNNING')",
            (self._now(),),
        )
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── CRUD ─────────────────────────────────────────────────────────────────
    async def find_duplicate(
        self, file_hash: str, schema_hash: str, org_id: Optional[str]
    ) -> Optional[dict]:
        """Return an existing non-failed job for the same file+schema+org, if any.

        FAILED jobs are excluded so a resubmission after a failure gets a fresh attempt.
        """
        cursor = await self._db.execute(
            "SELECT * FROM jobs WHERE file_hash = ? AND schema_hash = ? AND org_id IS ? "
            "AND status != 'FAILED' ORDER BY created_at DESC LIMIT 1",
            (file_hash, schema_hash, org_id),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def create_job(
        self,
        job_id: str,
        org_id: Optional[str],
        file_hash: str,
        schema_hash: str,
        schema_name: str,
        filename: str,
        file_path: str,
        schema_json: str,
    ) -> dict:
        now = self._now()
        await self._db.execute(
            "INSERT INTO jobs (job_id, org_id, file_hash, schema_hash, schema_name, "
            "filename, file_path, schema_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, org_id, file_hash, schema_hash, schema_name, filename, file_path, schema_json, now, now),
        )
        await self._db.commit()
        return await self.get_job(job_id)

    async def get_job(self, job_id: str) -> Optional[dict]:
        """Internal lookup (workers): no tenant scoping."""
        cursor = await self._db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_job_for_org(self, job_id: str, org_id: Optional[str]) -> Optional[dict]:
        """API lookup: a job is visible only to the organisation that created it. Requests without
        X-Organization-Id see only jobs created without one."""
        cursor = await self._db.execute(
            "SELECT * FROM jobs WHERE job_id = ? AND org_id IS ?", (job_id, org_id)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        stage: Optional[str] = None,
        result: Optional[dict] = None,
        error: Optional[dict] = None,
    ):
        # Never overwrite a terminal status
        job = await self.get_job(job_id)
        if job and job["status"] in (JobStatus.SUCCESS, JobStatus.PARTIAL, JobStatus.FAILED):
            return
        updates = ["status = ?", "updated_at = ?"]
        params = [status, self._now()]
        if stage is not None:
            updates.append("stage = ?")
            params.append(stage)
        if result is not None:
            updates.append("result = ?")
            params.append(json.dumps(result))
        if error is not None:
            updates.append("error = ?")
            params.append(json.dumps(error))
        params.append(job_id)
        await self._db.execute(
            f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?", params
        )
        await self._db.commit()

    async def get_pending_jobs(self) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM jobs WHERE status = 'PENDING' ORDER BY created_at"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

