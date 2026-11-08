"""Tests for FastAPI endpoints: /health, /jobs, /jobs/{id}."""
import json
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app, store


@pytest.fixture(autouse=True)
async def setup_db():
    await store.init()
    yield
    await store.close()


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_get_nonexistent_job():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/jobs/nonexistent-id-12345")
        assert resp.status_code == 404
        assert resp.json()["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_create_job_unsupported_file():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.exe", b"binarycontent", "application/octet-stream")}
        data = {"schema": json.dumps({"name": "test", "fields": {}})}
        resp = await client.post("/jobs", files=files, data=data)
        assert resp.status_code == 400
        assert resp.json()["code"] == "UNSUPPORTED_FILE_TYPE"


@pytest.mark.asyncio
async def test_create_job_invalid_schema():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.csv", b"a,b,c\n1,2,3", "text/csv")}
        data = {"schema": "not-valid-json{"}
        resp = await client.post("/jobs", files=files, data=data)
        assert resp.status_code == 400
        assert resp.json()["code"] == "INVALID_SCHEMA"


@pytest.mark.asyncio
async def test_create_job_and_poll():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        schema_json = json.dumps({
            "name": "simple",
            "fields": {
                "field1": {"name": "Field 1", "description": "Test field"}
            }
        })
        files = {"file": ("test.csv", b"field1\nvalue1", "text/csv")}
        resp = await client.post("/jobs", files=files, data={"schema": schema_json})
        assert resp.status_code == 202
        job_data = resp.json()
        assert "job_id" in job_data
        assert job_data["status"] in ("PENDING", "RUNNING", "SUCCESS", "FAILED")
        job_id = job_data["job_id"]

        # Poll job
        poll_resp = await client.get(f"/jobs/{job_id}")
        assert poll_resp.status_code == 200
        assert poll_resp.json()["job_id"] == job_id


@pytest.mark.asyncio
async def test_idempotent_duplicate_job():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        schema_json = json.dumps({
            "name": "simple_dedup",
            "fields": {
                "f": {"name": "F", "description": "Desc"}
            }
        })
        file_bytes = b"f\nhello_world"
        files1 = {"file": ("test_dedup.csv", file_bytes, "text/csv")}
        resp1 = await client.post("/jobs", files=files1, data={"schema": schema_json})
        assert resp1.status_code == 202
        job1 = resp1.json()

        files2 = {"file": ("test_dedup.csv", file_bytes, "text/csv")}
        resp2 = await client.post("/jobs", files=files2, data={"schema": schema_json})
        assert resp2.status_code == 202
        job2 = resp2.json()

        # Should return same job id
        assert job1["job_id"] == job2["job_id"]


@pytest.mark.asyncio
async def test_tenant_isolation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        schema_json = json.dumps({
            "name": "tenant_test",
            "fields": {"f": {"name": "F", "description": "Desc"}}
        })
        files = {"file": ("org_doc.csv", b"f\nsecret_tenant_data", "text/csv")}
        resp = await client.post(
            "/jobs",
            files=files,
            data={"schema": schema_json},
            headers={"X-Organization-Id": "org-alpha"},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Owner can access
        ok_resp = await client.get(f"/jobs/{job_id}", headers={"X-Organization-Id": "org-alpha"})
        assert ok_resp.status_code == 200

        # Different tenant gets 404 (not 403, per spec!)
        other_resp = await client.get(f"/jobs/{job_id}", headers={"X-Organization-Id": "org-beta"})
        assert other_resp.status_code == 404



@pytest.mark.asyncio
async def test_request_without_org_cannot_read_org_job():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        schema_json = json.dumps({"name": "iso2", "fields": {"f": {"name": "F", "description": "D"}}})
        files = {"file": ("o.csv", b"f\nscoped", "text/csv")}
        resp = await client.post("/jobs", files=files, data={"schema": schema_json},
                                 headers={"X-Organization-Id": "org-alpha"})
        job_id = resp.json()["job_id"]
        assert (await client.get(f"/jobs/{job_id}")).status_code == 404


@pytest.mark.asyncio
async def test_failed_job_is_not_reused_by_dedup():
    from app.models import JobStatus
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        schema_json = json.dumps({"name": "retry", "fields": {"f": {"name": "F", "description": "D"}}})
        files = {"file": ("r.csv", b"f\nretry_me", "text/csv")}
        job1 = (await client.post("/jobs", files=files, data={"schema": schema_json})).json()
        await store.update_status(job1["job_id"], JobStatus.FAILED, error={"code": "X", "message": "boom"})
        files = {"file": ("r.csv", b"f\nretry_me", "text/csv")}
        job2 = (await client.post("/jobs", files=files, data={"schema": schema_json})).json()
        assert job2["job_id"] != job1["job_id"]


@pytest.mark.asyncio
async def test_terminal_status_never_changes():
    from app.models import JobStatus
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        schema_json = json.dumps({"name": "term", "fields": {"f": {"name": "F", "description": "D"}}})
        files = {"file": ("t.csv", b"f\nterminal", "text/csv")}
        job = (await client.post("/jobs", files=files, data={"schema": schema_json})).json()
        await store.update_status(job["job_id"], JobStatus.SUCCESS, result={"entries": [], "metadata": {}})
        await store.update_status(job["job_id"], JobStatus.RUNNING, stage="X")
        await store.update_status(job["job_id"], JobStatus.FAILED, error={"code": "X", "message": "y"})
        assert (await client.get(f"/jobs/{job['job_id']}")).json()["status"] == "SUCCESS"
