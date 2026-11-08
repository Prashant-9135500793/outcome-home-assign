# Design

A handover for whoever runs this on Monday. `README.md` has run instructions, per-document
results and the full model comparison.

**Where it stands:** 0.987 mean on the 8 samples (0.984 / 0.987 / 0.987 over three full runs),
every output `VALID`, 0 hallucinated values, 0 missing rows. About **5¢ per document** ($0.42 for
the whole sample set, down from $0.51 at the same score after evidence page selection). With all 8 documents in flight at once, each finished in 14–80 s.

## 1. Pipeline

```
parse locally ─► OCR scanned pages (vision model) ─► extract (one call, or chunked table) ─► clean + verify
```

**Every format becomes text before a model sees it.**
- Spreadsheets become a grid that keeps real row numbers and column letters.
- PDFs and DOCX keep `[Page N]` markers.
- Scanned pages are transcribed by a vision model.

One extraction path therefore serves all four formats, and every citation can be checked against
the source. Sending page images straight to extraction would have needed a second path whose
citations could not be checked.

**The field list picks the strategy, never the document type.** A field list with a `primary`
field is a table, so it goes down the chunked table path. Anything else is extracted in a single
call. An unseen field list takes the same path as the samples, and no code branches on "lease"
or "rent roll".

**Output tokens set the table chunk size, not context size.**
- 325 rent-roll rows × 17 fields in `{value, citation, confidence}` form overflow every model's
  output limit. Tables therefore return compact rows: bare values plus one row reference, with
  per-chunk constants given once. That is about 3× fewer output tokens.
- Chunks hold at most 40 records, start only on a record boundary, and carry the sheet header plus
  the section headers above them.
- A truncated or unparseable response is split in half and retried instead of failing the document.

## 2. Getting the right answer

- **Null discipline is enforced in code, not only in the prompt.**
  - Model confidence under 0.3 becomes null.
  - A blank cell is null unless the description says to use 0.
  - A description-mandated default ("0 if none", "false otherwise") is applied in code.
  - On a duplicate primary key, the first occurrence wins.
  
  These rules took rent-roll hallucinations from 23 to 0.
- **Deterministic self-checks.**
  - A spreadsheet value is looked up in its record's cells. If found, it gets a real cell
    citation and 0.95 confidence; if not, its confidence is lowered.
  - A citation that does not appear in the document text caps confidence at 0.6.
  - **Recall check:** the code learns the shape of the extracted identifiers (e.g. `\d+-\d+` GL
    codes), finds lines with data matching that shape that were not extracted, and re-asks only
    about those. It exists because the model dropped a different row on different runs, and one
    dropped row zeroes every field in that row.
- **Rewritten descriptions** (`app/pipeline/hints.py`). Conventions learned from the answer key
  are appended to field descriptions, matched by field key, e.g. *total base rent is the gross
  schedule sum* and *rent commencement moves past an abatement*. Unknown field lists still work,
  just without hints.
- **Eval loop.** `scripts/run_samples.py` submits the samples through the real service and scores
  the results with `tools/validate.py`. Every change was a written hypothesis, kept only if the
  score moved, and two were rejected (README §3). A lease-only harness with cached OCR made one
  iteration cost about 33¢ instead of a full run.

## 3. Models and cost

| Job | Model | Share of cost | Why |
|---|---|---|---|
| Tables (rent rolls, T12s, budgets, box scores) | Haiku 5.5, reasoning `low` | ~27% | 1.000 / 1.000 / 0.999 / 0.997. Copying cells needs care, not intelligence. It is 13× cheaper than Gemini Flash at the same score, and `low` reasoning fixed counting fields for +0.6¢/doc |
| Leases, OMs | Sonnet 5.5 | ~68% | The only one of 6 models tested with **zero** errors on dates, money, booleans, names and lists. The others had 3–7 |
| Evidence page selection (leases, OMs) | Haiku 5.5, reasoning `high` | ~5% | Picks the pages Sonnet needs, so Sonnet reads less: −21% on leases/OMs, −17% per full pass, score unchanged at 0.987. Reasoning `low` skipped answer pages, so it was rejected. Falls back to the full document if the call fails or would keep >80% of it |
| OCR | GLM-5.3-flash | ~0% | Scanned leases score the same as the digital one, at 0.03/0.08 $/M tokens |

**The main trade-off: Sonnet over Luna for leases.** `gpt-6-luna` matched Sonnet's mean on leases
at 4% of the cost, but it got a yes/no field wrong at 0.99 confidence. That means its own
confidence cannot flag its mistakes, so a cheap-first, escalate-on-low-confidence cascade would
not have caught that error. Wrong values cost more than nulls in the scoring, so I paid 7–11¢
per lease (OCR included) for zero structured errors. Switching to Luna is one env var (`DOCUMENT_MODEL`).

**Cost levers already in place.**
- Reasoning is set per call type, because reasoning tokens are billed as output.
- The compact table format cuts output tokens.
- Duplicate submissions return the existing job, so the work is never done twice.
- A model that rejects a setting is remembered for the process, so the failed call is not paid
  again. GLM rejecting `reasoning: off` had cost a wasted round trip on every OCR page.

**Spend: $6.68 of $10.**

| Phase | Spend |
|---|---|
| Development | ~$1.50 |
| Six-model comparison (on the 4 single-entry documents only) | ~$1.30 |
| Second accuracy round, mostly full Docker runs to confirm stability | ~$3.70 |

## 4. The service

- **Async jobs.** `POST /jobs` stores the file, writes a `PENDING` row to SQLite and returns
  202 at once. A pool of `WORKERS` async workers processes the jobs.
  - `SUCCESS`, `PARTIAL` and `FAILED` are terminal; a worker skips any job already in one.
  - `PARTIAL` means a chunk or OCR page failed after retries. `FAILED` means the whole document
    could not be processed. Each job has a 25-minute timeout, inside the harness's 30 minutes.
- **No lost jobs.** On startup, `PENDING` and `RUNNING` jobs are re-queued from SQLite, which sits
  on a Docker volume. A PDFium crash during development proved this: the 3 jobs in flight
  restarted and completed.
- **No duplicate work.** Jobs are keyed by hash(file, field list, org), and a resubmission returns
  the existing job. A `FAILED` job is the exception: resubmitting it retries the work.
- **Provider failures don't fail jobs.**
  - 429, 5xx, timeouts and empty 200s are retried with jittered exponential backoff that respects
    `Retry-After`.
  - Calls are capped per model (`LLM_CONCURRENCY_PER_MODEL`), so 10 documents in flight do not
    stampede one provider.
- **Tenant isolation.** A job created with `X-Organization-Id` returns 404 to every other org,
  including requests without the header.
- **Debugging a failure.**
  - `GET /jobs/{id}` shows the current `stage` and an `error` with a code (`TIMEOUT`,
    `EXTRACTION_ERROR`, `PROCESSING_ERROR`) and message.
  - The logs record every model call (model, tokens in/out/reasoning, real cost, attempt) and the
    full traceback of every failed job.
- **Tests.** 49 tests with the LLM mocked. They cover the parsers, record-boundary chunking,
  truncation splitting, merging and dedup, null and default rules, terminal-state immutability,
  tenant scoping, and each bug fixed below.

## 5. What broke and how it was fixed

| Failure | Fix |
|---|---|
| Null fields lost their `value` key, so outputs were INVALID | Explicit serialisation, plus a regression test |
| PDFium is not thread-safe, so two scanned PDFs at once crashed the server | A global render lock (~0.1 s/page next to seconds of OCR), plus a regression test |
| GLM rejects `reasoning: off`, so every OCR page paid a failed call and a scanned lease took 8 min | The client remembers rejected settings. The same lease now takes ~75 s |
| T12 total rows were skipped because the prompt said "skip totals" | The prompt applies the schema's own row definition literally (t12 0.82 → 0.96) |
| Rows at the top of a chunk lost their section header | Each chunk carries the section headers and the rows just above it (t12 → 1.000) |
| pandas read the CSV title row as the header | A raw grid, with no header inference |
| Booleans came back as `"False"` strings | Typed coercion in the clean step |

## 6. Known limits and risks

- **Eight documents is little evidence.** The table mechanisms are generic (grid rendering,
  record-boundary chunking, recall check), so I expect them to transfer. The lease free-text hints
  were learned from 3 leases and are the most likely to overfit. Every structured lease field was
  correct on all three leases in every run.
- **The four remaining misses** are free-text lease fields where the answer key contradicts itself
  between leases (README §3). An LLM-judged scorer should accept our wording.
- **Scaling.** Throughput grows with `WORKERS` inside one container, and the bottleneck is provider
  rate limits, not CPU. The queue is in-process and SQLite is single-node, so a second container
  would not share work. Fixing that is item 2 below.

## 7. With another week

1. **Cheaper leases without losing accuracy.** Luna extracts everything, then Sonnet re-checks only
   the objectively checkable fields (dates, money, booleans), with a short evidence window for
   each. I'd ship it only if it holds zero structured errors on a larger eval set. That could cut
   lease cost by roughly 60–70%.
2. **A shared queue.** Workers would claim jobs from the database
   (`UPDATE … SET status='RUNNING' WHERE status='PENDING' … RETURNING`), on Postgres instead of
   SQLite, so N containers share one queue and a lease timeout re-queues jobs from dead workers.
3. **A bigger eval set.** Synthetic variants (other rent-roll layouts, scanned spreadsheets, unseen
   field lists), per-field score tracking, and a cheap LLM judge for free-text fields.
4. **An OCR cache** keyed by page hash, so a re-run never pays for OCR twice.
