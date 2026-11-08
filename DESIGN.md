# Design

A one-page handover. `README.md` has how to run it, the full results and the model comparison.

## The shape of the pipeline

**Parse locally, then send text to a model.** Every format becomes text before any model sees it.
Spreadsheets become a grid that keeps the real row numbers and column letters; scanned PDF pages
become text through a vision model. One extraction path then serves every format, every chunk
can cite `[Page N]` or a cell, and spreadsheet citations can be checked against real cells.
Sending page images straight to extraction would have needed a second path, and the citations
could not have been checked.

**The field list decides the strategy.** A field list with a `primary` field is a table, so
the service chunks it and extracts rows. Without one, the document is extracted in a single call.
Nothing depends on knowing the document type, so a field list we have never seen takes the same
path as the samples.

**Output tokens set the chunk size, not input tokens.** Every model has a context window of 1M
tokens, but 325 rent-roll rows × 17 fields in the full `{value, citation, confidence}` shape does
not fit in any output limit. Tables therefore return compact rows: bare values plus one row
reference, with header fields given once per chunk. Chunks are capped at 40 records. A response
cut off anyway is split in half and retried rather than failing the document.

**We check our own work deterministically.**
- Spreadsheet values are looked up in their record's cells: found means a cell citation and 0.95
  confidence, not found means lower confidence.
- A citation that does not appear in the document text caps the confidence at 0.6.
- The **recall check** learns the shape of the extracted identifiers, finds identifier-shaped lines
  with data that were not extracted, and re-checks only those. It exists because the model
  dropped a different row on different runs, and one missing row zeroes every field in that row.

**Null discipline in code as well as in the prompt.** A value under 0.3 confidence becomes null.
A blank cell is null unless the description says to use 0. When a unit appears twice, the first
occurrence wins.

## Which model does what, and why

| Job | Model | Why |
|---|---|---|
| Tables | Haiku 5.5, reasoning `low` | 1.000 on all four table samples. Costs a seventh of Gemini Flash; `low` reasoning fixed counting fields for +$0.006 per document |
| Leases, OMs | Sonnet 5.5 | Only one of 6 models tested with **zero** errors on dates, booleans, names and money (others had 3–7) |
| OCR | GLM-5.3-flash | Accurate enough that the scanned leases score the same as the digital one; effectively free; 21 pages in 33 s |

The big trade-off: **gpt-6-luna** matched Sonnet's mean score on leases at **4% of the cost**,
but it got a yes/no field wrong at 0.99 confidence. Accuracy carries more weight than cost, so
Sonnet stays at about 9¢ per lease. Luna is one env var away (`DOCUMENT_MODEL`).

**Spend:** $6.68 of $10 in total. A full sample pass costs about $0.50: roughly 65% Sonnet on
leases and OM, 30% Haiku on tables, and close to $0 for OCR. The six-model comparison cost about
$1.30. The second accuracy round (0.959 → 0.987) cost about $3.70, most of it on full Docker runs
used to confirm stability.

## What broke, and what we did

| Failure | Fix |
|---|---|
| Fields with a null value lost their `value` key (`exclude_none` in serialisation), so outputs were INVALID | Explicit `to_output()`; regression test |
| PDFium is not thread-safe; two scanned documents at once crashed the server | Global render lock (rendering is ~0.1 s per page); regression test |
| The crash showed recovery works: the 3 in-flight jobs restarted from SQLite and completed | – |
| GLM rejects `reasoning: off` (HTTP 400), so every OCR page paid a failed call; scanned leases took 8 min | Remember models that reject a setting; now about 75 s |
| t12 total rows were skipped because the prompt said "skip totals" while the schema says GL-coded totals are rows | Prompt applies the schema's row definition literally |
| Rows at the start of a chunk lost their section header, which sat in the previous chunk | Each chunk gets the section headers and rows just above it |
| pandas treated the CSV title row ("Box Score") as the column header | Raw CSV grid, no header inference |
| Model returned `"False"` strings for booleans | Coerced to JSON booleans |

Provider throttling and transient errors (429, 5xx, timeouts, 200 with no choices) are retried
with jittered backoff and never fail a job.

## Decisions where the brief is silent

- **Truth conventions learned from the samples** and written as field hints
  (`app/pipeline/hints.py`), e.g. total base rent is the *gross* schedule sum and rent
  commencement moves past an abatement. Hints match by field key, so unseen field lists still work.
- **PARTIAL** when a chunk or an OCR page fails after retries, or when nothing at all was found.
  **FAILED** only when the whole document could not be processed.
- **Tenant isolation:** a job created with `X-Organization-Id` is a 404 for everyone else,
  including requests without the header.
- **Dedup** returns the existing job for the same file + field list + org, except a FAILED one:
  resubmitting retries the work.
- **Possible answer-key inconsistency:** lease free-text answers mix verbatim quotes (`late_fee`)
  with paraphrases (`right_to_sublet`). We follow the per-field style through hints, but this
  is where the strict validator and an LLM judge will disagree most.

## How much to trust the 0.987

The tables (1.000 / 0.999 / 0.997 / 1.000) come from generic mechanisms (grid rendering,
record-boundary chunking, the recall check, literal row rules), so I expect them to carry over to
new documents. The lease score is more optimistic. The last +0.03 came from hints that teach the
answer key's *style* for free-text fields (verbatim spans, phrase patterns), learned on only 3
leases. They should help on new leases from the same pipeline, but 3 examples is little evidence,
and on free text the real LLM-judged scorer is lenient anyway. What I am confident in is that
every structured lease field (dates, money, booleans, lists) was correct on all three leases in
every run.

## With another week

1. **Free-text accuracy:** a cheap LLM-judge step in the eval loop to measure lease wording the
   way the real scorer does, then tune the hints against it rather than against exact match.
2. **Cost on leases:** a Luna first pass, with Sonnet re-checking only the structured fields
   (dates, money, booleans) instead of the whole document, which could cut lease cost about 70%.
3. **A bigger eval set:** 8 documents can't separate models that differ by one field. I'd make
   synthetic variants (other rent-roll layouts, scanned spreadsheets) and track scores per field.
4. **Service:** move the queue from in-process asyncio to the database (claim with
   `UPDATE … WHERE status='PENDING'`) so several containers can share one queue; add a
   per-job event log for debugging; add webhooks.
5. **OCR cache** keyed by page hash, so re-running a scanned document never pays for OCR twice.
