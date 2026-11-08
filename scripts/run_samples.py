#!/usr/bin/env python3
"""Eval loop: submit sample documents to a running service, collect results into outputs/,
then validate and score them with tools/validate.py.

Usage:
  python scripts/run_samples.py                       # all documents in sample/manifest.csv
  python scripts/run_samples.py boxscore_001 t12_001  # only these documents
  python scripts/run_samples.py --base-url http://localhost:8080 --concurrency 4

The service deduplicates identical (file, schema) submissions, so re-running without changes
returns the cached job. Pass --bust to append a nonce to the schema and force a fresh run.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample"
TERMINAL = {"SUCCESS", "PARTIAL", "FAILED"}


async def run_one(client: httpx.AsyncClient, doc_id: str, file_name: str, schema_name: str,
                  out_dir: Path, bust: bool, timeout_s: int) -> dict:
    doc_path = SAMPLE / "sample-data-documents" / file_name
    schema = json.loads((SAMPLE / "schemas" / f"{schema_name}.json").read_text())
    if bust:
        schema["_nonce"] = uuid.uuid4().hex  # changes the schema hash; ignored by extraction
    t0 = time.time()
    with open(doc_path, "rb") as fh:
        r = await client.post("/jobs", files={"file": (file_name, fh.read())},
                              data={"schema": json.dumps(schema)})
    if r.status_code != 202:
        return {"doc": doc_id, "status": f"HTTP {r.status_code}", "detail": r.text[:200]}
    job = r.json()
    job_id = job["job_id"]
    print(f"  submitted {doc_id:15s} -> {job_id}")

    while job["status"] not in TERMINAL:
        if time.time() - t0 > timeout_s:
            return {"doc": doc_id, "status": "CLIENT_TIMEOUT", "job_id": job_id}
        await asyncio.sleep(2)
        job = (await client.get(f"/jobs/{job_id}")).json()

    elapsed = time.time() - t0
    summary = {"doc": doc_id, "status": job["status"], "job_id": job_id, "seconds": round(elapsed, 1)}
    if job.get("result"):
        result = job["result"]
        result["schema"] = schema_name
        (out_dir / f"{doc_id}.json").write_text(json.dumps(result, indent=2))
        md = result.get("metadata", {})
        summary.update(rows=len(result.get("entries", [])), calls=md.get("model_calls"),
                       cost=md.get("cost_usd"))
    if job.get("error"):
        summary["error"] = job["error"]
    return summary


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="*", help="document ids (default: all in manifest)")
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--bust", action="store_true", help="bypass the service's duplicate-job cache")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    with open(SAMPLE / "manifest.csv") as fh:
        manifest = list(csv.DictReader(fh))
    if args.docs:
        unknown = set(args.docs) - {m["document_id"] for m in manifest}
        if unknown:
            print(f"unknown document ids: {sorted(unknown)}")
            return 2
        manifest = [m for m in manifest if m["document_id"] in args.docs]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=60) as client:
        try:
            (await client.get("/health")).raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"service not reachable at {args.base_url}: {e}")
            return 2

        async def guarded(m):
            async with sem:
                return await run_one(client, m["document_id"], m["file"], m["schema_name"],
                                     out_dir, args.bust, args.timeout)

        print(f"Running {len(manifest)} document(s) against {args.base_url}")
        summaries = await asyncio.gather(*(guarded(m) for m in manifest))

    print("\n=== Run summary")
    total_cost = 0.0
    for s in summaries:
        total_cost += s.get("cost") or 0
        print(f"  {s['doc']:15s} {s['status']:8s} rows={s.get('rows', '-'):>4} calls={s.get('calls', '-'):>3} "
              f"cost=${(s.get('cost') or 0):.4f} time={s.get('seconds', '-')}s"
              + (f"  error={s['error']}" if s.get("error") else ""))
    print(f"  total reported cost: ${total_cost:.4f}")

    if not args.no_validate:
        print("\n=== Validation and score")
        targets = [out_dir / f"{s['doc']}.json" for s in summaries if (out_dir / f"{s['doc']}.json").exists()]
        for t in targets:
            m = next(m for m in manifest if m["document_id"] == t.stem)
            subprocess.run([sys.executable, str(ROOT / "tools" / "validate.py"), str(t),
                            "--schema", str(SAMPLE / "schemas" / f"{m['schema_name']}.json"),
                            "--truth", str(SAMPLE / "sample-data-output-validated" / t.name)])
        if not args.docs:
            subprocess.run([sys.executable, str(ROOT / "tools" / "validate.py"), str(out_dir),
                            "--schema-dir", str(SAMPLE / "schemas"),
                            "--truth-dir", str(SAMPLE / "sample-data-output-validated"), "-q"])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
