"""FastAPI application: POST /jobs, GET /jobs/{job_id}, GET /health."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import MAX_FILE_SIZE, UPLOAD_DIR, PORT
from app.database import JobStore
from app.models import JobStatus
from app.worker import WorkerPool

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Globals ──────────────────────────────────────────────────────────────────
store = JobStore()
worker_pool = WorkerPool(store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    await store.init()
    await worker_pool.start()
    logger.info("Service ready")
    yield
    await worker_pool.stop()
    await store.close()


app = FastAPI(title="Document Extraction Service", version="1.0.0", lifespan=lifespan)


# ── Helpers ──────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv"}


def _file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _schema_hash(schema_str: str) -> str:
    return hashlib.sha256(schema_str.encode()).hexdigest()


def _format_job_response(job: dict) -> dict:
    """Convert a DB row into the API response shape."""
    resp = {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
    if job.get("stage"):
        resp["stage"] = job["stage"]
    if job["status"] in (JobStatus.SUCCESS, JobStatus.PARTIAL) and job.get("result"):
        resp["result"] = json.loads(job["result"]) if isinstance(job["result"], str) else job["result"]
    if job["status"] == JobStatus.FAILED and job.get("error"):
        resp["error"] = json.loads(job["error"]) if isinstance(job["error"], str) else job["error"]
    return resp


# ── GET /health ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── POST /jobs ───────────────────────────────────────────────────────────────
@app.post("/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    schema_str: str = Form(..., alias="schema"),
    webhook_url: str | None = Form(None),
    x_organization_id: str | None = Header(None),
):
    # Validate file type
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"code": "UNSUPPORTED_FILE_TYPE", "message": f"File type {ext} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"},
        )

    # Validate schema JSON
    try:
        schema_data = json.loads(schema_str)
        if "fields" not in schema_data:
            raise ValueError("Schema must contain a 'fields' key")
    except (json.JSONDecodeError, ValueError) as e:
        return JSONResponse(
            status_code=400,
            content={"code": "INVALID_SCHEMA", "message": f"Schema is not valid JSON: {e}"},
        )

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(status_code=413, content={"code": "FILE_TOO_LARGE", "message": "File exceeds 50 MB"})
    if len(content) == 0:
        return JSONResponse(status_code=400, content={"code": "EMPTY_FILE", "message": "File is empty"})

    # Compute hashes for deduplication
    fh = _file_hash(content)
    sh = _schema_hash(schema_str)

    # Check for duplicate
    existing = await store.find_duplicate(fh, sh, x_organization_id)
    if existing:
        logger.info(f"Duplicate submission, returning existing job {existing['job_id']}")
        return JSONResponse(status_code=202, content=_format_job_response(existing))

    # Save file to disk
    job_id = str(uuid.uuid4())
    file_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, filename)
    with open(file_path, "wb") as f:
        f.write(content)

    # Create job in DB
    job = await store.create_job(
        job_id=job_id,
        org_id=x_organization_id,
        file_hash=fh,
        schema_hash=sh,
        schema_name=schema_data.get("name", "unknown"),
        filename=filename,
        file_path=file_path,
        schema_json=schema_str,
    )

    # Submit to worker pool
    await worker_pool.submit(job_id)

    return JSONResponse(status_code=202, content=_format_job_response(job))


# ── GET /jobs/{job_id} ───────────────────────────────────────────────────────
@app.get("/jobs/{job_id}")
async def get_job(job_id: str, x_organization_id: str | None = Header(None)):
    job = await store.get_job(job_id, org_id=x_organization_id)
    if not job:
        return JSONResponse(status_code=404, content={"code": "NOT_FOUND", "message": "Job not found"})
    return JSONResponse(status_code=200, content=_format_job_response(job))


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)

