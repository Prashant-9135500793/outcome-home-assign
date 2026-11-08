# Document Extraction Service

A service that takes a commercial real estate document (PDF, DOCX, XLSX or CSV) and a field list,
and returns the extracted values as JSON in the format defined in `contract/`.

**Role applied for:** _TODO: `aiml` or `software`_

**Sample-set score (`tools/validate.py`):** **0.959 mean** over the 8 sample documents.
Every output is `VALID`, with **0 hallucinations** and **0 missing rows**. A full pass costs about
**$0.50**, roughly 6¢ per document.

---

## 1. Running it

### With Docker (how it is meant to run)

```bash
cp .env.example .env          # then put your OpenRouter key in LLM_API_KEY
docker compose up --build     # service on http://localhost:8080
curl localhost:8080/health    # {"status":"ok"}  (ready ~3 s after start)
```

Nothing else needs installing. The job database and uploaded files live in a named Docker volume
(`job-data`), so queued jobs survive a container restart.

### Without Docker (development)

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8080
```

### Submitting a document

```bash
# 1. submit: returns 202 immediately with a job_id
curl -F file=@sample/sample-data-documents/boxscore_001.csv \
     -F schema=@sample/schemas/boxscore.json \
     localhost:8080/jobs

# 2. poll until status is SUCCESS / PARTIAL / FAILED
curl localhost:8080/jobs/<job_id>
```

An optional `X-Organization-Id` header scopes a job to one tenant. Other tenants, and requests
without the header, get a 404 for it.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible gateway (all model calls go through it) |
| `LLM_API_KEY` | – | Key sent as `Authorization: Bearer` |
| `WORKERS` | `10` (compose) | Documents processed in parallel |
| `TABLE_MODEL` | `anthropic/claude-haiku-5.5` | Model for table documents |
| `DOCUMENT_MODEL` | `anthropic/claude-sonnet-5.5` | Model for single-entry documents (leases, OMs) |
| `OCR_MODEL` | `z-ai/glm-5.3-flash` | Vision model that transcribes scanned pages |
| `TABLE_REASONING` / `DOCUMENT_REASONING` / `OCR_REASONING` | `low` / `default` / `low` | Reasoning effort per call type (`off`, `low`, `medium`, `high`, `default`) |
| `LLM_CONCURRENCY_PER_MODEL` | `16` | Parallel calls per model across all jobs |

Only models on the brief's allowed list are accepted. The service refuses to start with any other.

### Tests and the eval loop

```bash
.venv/bin/pytest -q                                   # 43 tests, no API calls (LLM is mocked)

# with the service running:
python scripts/run_samples.py                         # all 8 samples -> outputs/, then scores them
python scripts/run_samples.py t12_001 budget_001      # just these
python scripts/run_samples.py --bust                  # bypass the duplicate-job cache

python tools/validate.py outputs/ --schema-dir sample/schemas --truth-dir sample/sample-data-output-validated
```

---

## 2. How it works

```
POST /jobs ──► validate type + schema ──► hash(file, schema, org) ──► duplicate? return existing job
                                               │ new
                                               ▼
                        save file, insert job row (SQLite, PENDING) ──► 202 { job_id }
                                               │
                     worker pool (WORKERS async workers, jobs re-queued from the DB on restart)
                                               ▼
   1. PARSE    PDF: pdfplumber text + tables per page; pages with no text layer are marked scanned
               XLSX/CSV: grid with real row numbers and column letters (empty columns dropped)
               DOCX: paragraphs and tables in document order, plus headers/footers
   2. OCR      scanned pages -> JPEG -> vision model transcription (parallel), keeps [Page N] markers
   3. EXTRACT  field list has a primary field?  ── no ──► single-entry path (one call to DOCUMENT_MODEL)
                                                 └ yes ─► table path (chunked, TABLE_MODEL)
   4. CLEAN    type coercion, null discipline, citation grounding, confidence
               ──► SUCCESS / PARTIAL / FAILED  (terminal statuses never change)
```

### Table documents (rent rolls, T12s, box scores, budgets)
- **Chunking on record boundaries.** A rent roll unit spans ~9 sheet rows (a unit line plus its
  charge lines). Chunks only start on a row whose first column is filled, so a record is never cut
  in half. Each chunk is capped by size and by record count; the cap bounds the output tokens.
- **Context for every chunk.** Each chunk also sees the sheet's header rows, the section-header
  rows above it and the last few rows before it. Without this, rows at the start of a chunk lost
  their `section` value (t12 went from 0.95 to 1.00 on that field).
- **Compact output.** Fields that are the same for every row (detected from the descriptions)
  are returned once per chunk. Rows come back as bare values plus one row reference, about 3× fewer
  output tokens than a `{value, citation, confidence}` object per cell.
- **Parallel and self-healing.** Chunks run concurrently. A response cut off at the output limit
  is split in half and retried, and unparseable JSON is retried once and then split.
- **Merging.** Rows are de-duplicated by the primary field; the first occurrence in document
  order wins. Document-level fields are decided by a vote across chunks.
- **Recall check.** The service learns the shape of the identifiers it extracted (e.g. `\d+-\d+`
  for GL codes) and scans the document for matching lines that carry data but were not extracted.
  Only those lines go back to the model for a targeted re-check. This caught a row the model
  dropped at random in the budget PDF.
- **Grounded citations.** For spreadsheets every value is looked up in its record's cells, so the
  citation is the actual cell (`[Willow Creek Commons!I10] 2842`). A value that cannot be found
  there (a computed sum, for instance) gets a lower confidence.

### Single-entry documents (leases, offering memoranda)
- The whole document goes to `DOCUMENT_MODEL` in one call (up to ~100K tokens). A larger
  document is split into parts that run in parallel and are merged by confidence.
- Each field returns `{value, citation, confidence}`. A value under 0.3 confidence becomes null,
  and a citation that cannot be found in the document text caps the confidence at 0.6.
- **Field hints** (`app/pipeline/hints.py`) extend the descriptions of known fields with what
  the eval loop taught us, e.g. *total base rent is the gross schedule sum, not net of abatement*.
  They match by field key, so a field list we have never seen still works, just without hints.

### Reliability
- Retries with jittered backoff on 429, 5xx, timeouts and empty 200 responses.
- If a model rejects a request setting (e.g. GLM refuses `reasoning: off`), the service
  remembers that for the rest of the process instead of paying a failed call every time.
- A crashed or restarted worker does not lose jobs: `PENDING`/`RUNNING` jobs are re-queued from
  SQLite on startup. This happened by accident during development and the jobs completed.
- Submitting the same file and field list twice returns the same job. A `FAILED` job is not
  reused, so resubmitting retries the work.
- PDFium (the page renderer) is not thread-safe, and concurrent use crashed the server. Rendering
  is now behind a lock; it takes about 0.1 s per page next to seconds for the vision call.

---

## 3. Results

### Final scores on the sample set

All 8 documents submitted at once to `docker compose up` (`WORKERS=10`):

| Document | Kind | Score | Rows | Model calls | Cost | Latency |
|---|---|---|---|---|---|---|
| boxscore_001 | CSV, 1 table | **1.000** | 10/10 | 1 | $0.0020 | 16 s |
| t12_001 | XLSX, 12-month statement | **1.000** | 119/119 | 4 | $0.0152 | 31 s |
| rentroll_001 | XLSX, 3,000-row block layout | **1.000** | 325/325 | 12 | $0.0846 | 56 s |
| om_001 | PDF, 22 pages | **1.000** | 1/1 | 2 | $0.0529 | 27 s |
| budget_001 | PDF, 8 pages | **0.999** | 55/55 | 3 | $0.0120 | 33 s |
| lease_001 | scanned PDF, 25 pages | **0.900** | 1/1 | 26 | $0.1118 | 70 s |
| lease_002 | DOCX | **0.900** | 1/1 | 1 | $0.0909 | 31 s |
| lease_003 | scanned PDF, 21 pages | **0.875** | 1/1 | 22 | $0.1291 | 80 s |
| **Mean / total** | | **0.959** | | 71 | **$0.4985** | |

Hallucinations: **0**. Missing or extra rows: **0**.

**Where the remaining points go.** Every number, date, boolean and name on the leases is correct.
The lease points lost are free-text fields such as `late_fee`, `holdover_rent` and
`right_to_sublet`. There the validator needs an exact match after normalisation, and our wording
differs slightly while meaning the same thing. Example:

> ours: *"only with Landlord's prior written consent"* · truth: *"with the prior written consent of Landlord"*

Every model we tried fails these the same way. The brief says the real scorer uses an LLM judge
for free text, so the true lease accuracy is likely higher than the table shows.

### How the eval loop moved the score

| Change | Effect |
|---|---|
| First working run (Gemini Flash for tables) | boxscore 1.000, but $0.018 for a 10-row CSV |
| Switched tables to Haiku, reasoning off | Same score, **13× cheaper** ($0.0014) |
| Fixed: null fields lost their `value` key | budget INVALID → VALID |
| Prompt: apply the row definition literally (GL-coded totals are rows), copy identifiers in full | t12 0.821 → 0.962 |
| Table reasoning `low` (cell counting) | t12 0.962 → 0.989 (+$0.006) |
| Header and preceding-row context for each chunk | t12 0.989 → 1.000 |
| Lease field hints (rent commencement after abatement, gross total rent, short phrases) | lease_002 0.775 → 0.900 |
| Rent roll: first occurrence wins, blank cell = null | rentroll 0.993 → 1.000 (23 hallucinations → 0) |
| Recall check | Recovers rows the model drops at random (budget 0.981 → 0.999) |
| Fixed: failed reasoning round trip on every OCR page | Scanned lease 8 min → about 75 s |

### Choosing the model for leases and OMs

All six candidates ran on the 4 single-entry documents. OCR (GLM) and tables (Haiku) were the
same in every run. "Structured errors" counts fields that can be checked objectively: dates,
booleans, names, numbers and lists. The free-text misses above are excluded because every model
has them.

| `DOCUMENT_MODEL` | Mean score | Structured errors | Cost (4 docs) |
|---|---|---|---|
| **anthropic/claude-sonnet-5.5** (chosen) | 0.913 | **0** | $0.368 |
| google/gemini-3.1-pro-preview | 0.900 | 3 | $0.557 |
| openai/gpt-6.1-sol | 0.900 | 6 | $0.274 |
| openai/gpt-6-luna | 0.910 | 3 | $0.015 |
| anthropic/claude-haiku-5.5 | 0.892 | 3 | $0.022 |
| deepseek/deepseek-v3.2 | 0.862 | 7 | $0.023 |

Sonnet was the only model with no structured errors. Luna came close on the mean at 4% of the
cost, but its misses included a yes/no field (`early_termination_right`) it got wrong at 0.99
confidence, so its confidence cannot be used to catch its mistakes. Accuracy outweighs cost in
the scoring, so leases and OMs use Sonnet (about 9¢ per lease). Swapping models is one env var
(`DOCUMENT_MODEL=openai/gpt-6-luna`) for anyone who wants the cheaper trade-off.

**Model prices** (OpenRouter, $ per 1M input / output tokens): Haiku 0.10/0.50 · Luna 0.10/0.50 ·
GLM-5.3-flash 0.03/0.08 · DeepSeek 0.26/0.38 · Gemini Flash 0.75/3.75 · Sonnet 2/10 ·
GPT-6.1-sol 2/10 · Gemini Pro 2/12.

### Budget

About **$3.00 of the $10** was spent across development, the model comparison and the final
runs. Each full sample pass costs about $0.50.

---

## 4. Repository layout

```
app/
  main.py              API: POST /jobs, GET /jobs/{id}, GET /health
  worker.py            async worker pool, job lifecycle, timeouts
  database.py          SQLite job store (dedup, tenant scoping, crash recovery)
  config.py            env configuration and model routing
  llm/client.py        OpenRouter client: retries, cost, reasoning control
  pipeline/
    parser.py          PDF / DOCX / XLSX / CSV -> text and grids; page rendering
    ocr.py             scanned pages -> vision model transcription
    extractor.py       chunking, extraction, recall check, merging, grounding
    prompts.py         prompt templates
    hints.py           field-description hints learned from the eval loop
scripts/run_samples.py eval loop: submit samples, write outputs/, score
tests/                 43 tests (parsers, chunking, truncation recovery, merging, API)
outputs/               service results for every sample document
contract/, tools/      interface contract and validator (provided)
```
