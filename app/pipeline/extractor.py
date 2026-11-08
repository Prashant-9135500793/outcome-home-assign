"""Core extraction: parse -> (OCR) -> LLM extraction -> merge -> clean/ground.

Two paths, chosen by the field list:
  - Single-entry schemas (no primary field): the whole document goes to DOCUMENT_MODEL in one
    call when it fits; otherwise parallel chunks merged by confidence.
  - Table schemas (one primary field): the document is split into chunks on row-record
    boundaries (spreadsheets) or page/line boundaries (PDF/DOCX), every chunk is extracted in
    parallel by TABLE_MODEL, a truncated or unparseable chunk is split in half and retried, and
    rows are merged and de-duplicated by the primary key. Document-level fields are voted
    across chunks so every row carries the same value.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from app import config
from app.llm.client import UsageStats, chat_completion
from app.models import ExtractionSchema, FieldResult
from app.pipeline.hints import apply_hints
from app.pipeline.ocr import ocr_scanned_pages
from app.pipeline.parser import (
    join_pages,
    parse_document,
    render_grid_header,
    render_grid_row,
)
from app.pipeline.selector import select_text
from app.pipeline.prompts import (
    build_single_entry_prompt,
    build_table_chunk_prompt,
    doc_level_keys,
    primary_key,
)

logger = logging.getLogger(__name__)

MAX_SPLIT_DEPTH = 4
GRID_CONTEXT_ROWS = 12          # leading rows of a sheet repeated as context in every chunk
GRID_PRECEDING_ROWS = 8         # rows immediately above a chunk, repeated as context
GRID_PRECEDING_LABELS = 6       # header-like rows further above a chunk, repeated as context
SMALL_SHEET_CONTEXT_CHARS = 3000
TEXT_HEAD_CONTEXT_CHARS = 1500
TEXT_TAIL_CONTEXT_CHARS = 800


@dataclass
class ExtractionOutcome:
    entries: list[dict[str, FieldResult]]
    partial: bool = False
    warnings: list[str] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# JSON handling
# ═════════════════════════════════════════════════════════════════════════════
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_response(text: str) -> dict:
    """Parse the first JSON object in a model response (tolerates fences and surrounding prose)."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from response: {text[:300]!r}")


# ═════════════════════════════════════════════════════════════════════════════
# Value cleaning
# ═════════════════════════════════════════════════════════════════════════════
_NUMERIC_HINTS = ("number", "amount", "rent", "cost", "price", "sq", "square", "units", "count",
                  "months", "rate", "ratio", "variance", "total", "budget", "deposit", "fee",
                  "acres", "parking", "noi", "rsf", "spaces", "stories", "occupancy", "balance",
                  "charges", "percent", "pct", "share", "year", "avg", "avail")
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y",
                 "%d %b %Y", "%Y/%m/%d", "%m-%d-%Y", "%B %d %Y", "%b. %d, %Y")
_NULL_STRINGS = {"", "n/a", "na", "none", "null", "-", "--", "not stated", "not specified", "unknown"}
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
_BOOL_STRINGS = {"true": True, "false": False, "yes": True, "no": False}


def _keeps_text(desc: str) -> bool:
    """Fields whose text must not be turned into a number/date (codes, IDs, verbatim text)."""
    d = desc.lower()
    if "as a string" in d or "verbatim" in d or "identifier" in d or re.search(r"\bcode\b", d):
        return True
    return "exactly as" in d and "number" not in d and "value of the" not in d


def try_parse_number(s: str) -> Optional[float]:
    t = s.strip().replace(",", "").replace("$", "").replace("€", "").replace("£", "").strip()
    neg = t.startswith("(") and t.endswith(")")
    if neg:
        t = t[1:-1].strip()
    if t.endswith("%"):
        t = t[:-1].strip()
    if not _NUM_RE.match(t):
        return None
    x = float(t)
    return -x if neg else x


def try_parse_date(val: str) -> Optional[str]:
    v = val.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[T ]\d{2}:\d{2}", v)
    if m:
        return m.group(1)
    return None


_ABSENT_ZERO_RE = re.compile(r"\b0 if (none|no\b|blank|empty|absent|not|there is no|the \w+ (has|have) no)", re.I)
_ABSENT_FALSE_RE = re.compile(r"\bfalse otherwise\b", re.I)
_ABSENT_LIST_RE = re.compile(r"\bempty list if\b", re.I)


def absent_default(field_desc: str) -> Any:
    """The value a description says to use when the thing is absent ("0 if none", "false otherwise",
    "Empty list if none"). Returning it for a null is following the field list, not guessing."""
    d = field_desc or ""
    if _ABSENT_ZERO_RE.search(d):
        return 0
    if _ABSENT_FALSE_RE.search(d):
        return False
    if _ABSENT_LIST_RE.search(d):
        return []
    return None


def coerce_value(value: Any, field_key: str, field_desc: str) -> Any:
    """Normalise a model value: strip null-ish strings, numbers out of money strings, ISO dates."""
    if value is None:
        return None
    if isinstance(value, dict):  # model wrapped the value one level too deep
        value = value.get("value")
        if value is None:
            return None
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return int(value)
    if not isinstance(value, str):
        return value

    s = value.strip()
    if s.lower() in _NULL_STRINGS:
        return None
    desc = (field_desc or "").lower()
    key = field_key.lower()

    # Boolean fields ("true if ..., false otherwise") must be JSON booleans, not "True"/"False"
    if re.search(r"\btrue\b.*\bfalse\b|\bwhether\b|\byes/no\b", desc) and s.lower() in _BOOL_STRINGS:
        return _BOOL_STRINGS[s.lower()]

    if not _keeps_text(desc):
        numeric_field = any(h in desc or h in key for h in _NUMERIC_HINTS)
        if numeric_field:
            num = try_parse_number(s)
            if num is not None:
                return int(num) if num == int(num) and "." not in s else num
        if "date" in key or "yyyy-mm-dd" in desc or re.search(r"\bdate\b", desc):
            d = try_parse_date(s)
            if d:
                return d
    return s


# ═════════════════════════════════════════════════════════════════════════════
# Matching (used for grounding citations and merging)
# ═════════════════════════════════════════════════════════════════════════════
def norm_key(v: Any) -> Any:
    """Normalise a value for equality comparisons (row identity, voting)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    s = str(v).strip()
    n = try_parse_number(s)
    if n is not None:
        return round(n, 4)
    d = try_parse_date(s)
    if d:
        return d
    return re.sub(r"\s+", " ", s).casefold()


def values_match(value: Any, cell_text: str) -> bool:
    if value is None or isinstance(value, (list, dict)):
        return False
    if isinstance(value, bool):
        return cell_text.strip().lower() in (("true", "yes", "y") if value else ("false", "no", "n"))
    if isinstance(value, (int, float)):
        n = try_parse_number(cell_text)
        # Cells hold exact values; only allow rounding noise (e.g. 2842.0049 printed as 2842)
        return n is not None and abs(n - float(value)) <= 0.005 + abs(float(value)) * 1e-9
    sv = str(value).strip()
    d1 = try_parse_date(sv)
    if d1:
        return try_parse_date(cell_text) == d1
    return norm_key(sv) == norm_key(cell_text)


_PAGE_RE = re.compile(r"\[Page (\d+)\]")


def pages_in(citation: str) -> Optional[list[int]]:
    pages = sorted({int(p) for p in _PAGE_RE.findall(citation or "")})
    return pages or None


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def citation_supported(citation: str, doc_text_squashed: str) -> bool:
    """True if the snippet part of a citation (after the [location] prefix) occurs in the document."""
    snippet = re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", citation or "").strip()
    sq = _squash(snippet)
    if len(sq) < 4:
        return True  # too short to check meaningfully
    if sq in doc_text_squashed:
        return True
    # Allow small edits / ellipses: check the first and last 24 characters separately
    return len(sq) > 48 and sq[:24] in doc_text_squashed and sq[-24:] in doc_text_squashed


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
StageCallback = Optional[Callable[[str], Awaitable[None]]]


async def extract_document(
    file_path: str,
    filename: str,
    schema: ExtractionSchema,
    usage_stats: UsageStats,
    update_stage: StageCallback = None,
) -> ExtractionOutcome:
    async def stage(name: str):
        if update_stage:
            await update_stage(name)

    await stage("PARSING")
    # Parsing is CPU-bound and synchronous; keep it off the event loop so the API stays responsive
    doc = await asyncio.to_thread(parse_document, file_path, filename)
    logger.info(f"Parsed {filename}: type={doc['type']} chars={len(doc['text'])} pages={len(doc['pages'])} "
                f"scanned={len(doc.get('scanned_pages') or [])}")

    warnings: list[str] = []
    if doc.get("scanned_pages"):
        await stage("OCR")
        await ocr_scanned_pages(doc, file_path, usage_stats)
        if doc.get("ocr_failed_pages"):
            warnings.append(f"OCR failed on pages {doc['ocr_failed_pages']}")

    await stage("EXTRACTING")
    schema = apply_hints(schema)
    if primary_key(schema):
        outcome = await _extract_table(schema, doc, usage_stats)
    else:
        outcome = await _extract_single(schema, doc, usage_stats)
    outcome.warnings = warnings + outcome.warnings
    outcome.partial = outcome.partial or bool(doc.get("ocr_failed_pages"))

    if not outcome.entries:
        outcome.entries = [{k: FieldResult(value=None, citation="", confidence=0.0) for k in schema.fields}]
        outcome.partial = True
        outcome.warnings.append("no rows extracted")
    logger.info(f"Extraction complete for {filename}: {len(outcome.entries)} entries, partial={outcome.partial}")
    return outcome


# ═════════════════════════════════════════════════════════════════════════════
# Single-entry documents
# ═════════════════════════════════════════════════════════════════════════════
def _split_pages_for_single(pages: list[dict], max_chars: int) -> list[str]:
    chunks, cur, size = [], [], 0
    for p in pages:
        t = f"[Page {p['page']}]\n{p['text']}"
        if cur and size + len(t) > max_chars:
            chunks.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(t)
        size += len(t)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


async def _single_call(schema, text, doc_type, usage, part_note="") -> dict:
    system, user = build_single_entry_prompt(schema, text, doc_type, part_note)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err: Exception | None = None
    for attempt in range(2):
        resp = await chat_completion(
            model=config.DOCUMENT_MODEL, messages=messages, usage_stats=usage,
            max_tokens=config.MAX_OUTPUT_TOKENS, response_format={"type": "json_object"},
            reasoning=config.DOCUMENT_REASONING,
        )
        try:
            parsed = parse_json_response(resp.content)
        except ValueError as e:
            last_err = e
            logger.warning(f"Unparseable single-entry response (attempt {attempt + 1}, truncated={resp.truncated})")
            continue
        fields = parsed.get("fields")
        if not isinstance(fields, dict):
            entries = parsed.get("entries")
            fields = entries[0] if isinstance(entries, list) and entries and isinstance(entries[0], dict) else parsed
        return fields
    raise ValueError(f"model returned unusable output twice: {last_err}")


async def _extract_single(schema: ExtractionSchema, doc: dict, usage: UsageStats) -> ExtractionOutcome:
    text = doc["text"]
    doc_type = doc["type"]
    partial = False
    warnings: list[str] = []

    if len(text) <= config.SINGLE_CALL_MAX_CHARS:
        # Cost: let a cheap model pick the evidence pages so DOCUMENT_MODEL reads fewer tokens
        model_text, sel = await select_text(schema, doc, usage, parse_json_response)
        note = ""
        if sel["selection"] != "full":
            note = ("NOTE: only the parts of the document relevant to the fields are included below "
                    "(marked excerpts are the edges of omitted pages). If a value is not stated in this text, return null.\n\n")
        raw_fields = await _single_call(schema, model_text, doc_type, usage, note)
    else:
        parts = _split_pages_for_single(doc["pages"], config.SINGLE_CALL_MAX_CHARS // 2) \
            if len(doc["pages"]) > 1 else \
            [text[i:i + config.SINGLE_CALL_MAX_CHARS // 2] for i in range(0, len(text), config.SINGLE_CALL_MAX_CHARS // 2)]
        logger.info(f"Single-entry document too large for one call; {len(parts)} parallel parts")
        note = "NOTE: this is part {i} of {n} of a longer document. Return null for fields not present in this part.\n\n"
        results = await asyncio.gather(
            *(_single_call(schema, p, doc_type, usage, note.format(i=i + 1, n=len(parts))) for i, p in enumerate(parts)),
            return_exceptions=True,
        )
        raw_fields = {}
        for r in results:
            if isinstance(r, Exception):
                partial = True
                warnings.append(f"part failed: {r}")
                continue
            for k, v in r.items():
                if k not in schema.fields or not isinstance(v, dict):
                    continue
                cur = raw_fields.get(k)
                if v.get("value") in (None, [], "") and cur is not None:
                    continue
                if cur is None or cur.get("value") in (None, [], "") or \
                        float(v.get("confidence") or 0) > float(cur.get("confidence") or 0):
                    raw_fields[k] = v
        if not raw_fields:
            raise RuntimeError("all document parts failed: " + "; ".join(warnings))

    squashed = _squash(text)
    entry: dict[str, FieldResult] = {}
    for key, fdef in schema.fields.items():
        raw = raw_fields.get(key)
        if isinstance(raw, dict):
            value, citation, conf = raw.get("value"), str(raw.get("citation") or ""), raw.get("confidence", 0.7)
        else:
            value, citation, conf = raw, "", 0.6 if raw is not None else 0.0
        entry[key] = _finalize_field(key, fdef.description, value, citation, conf, squashed, doc_type)
    return ExtractionOutcome([entry], partial, warnings)


def _finalize_field(key, desc, value, citation, conf, squashed_text, doc_type,
                    source_locations=None) -> FieldResult:
    if isinstance(value, list):
        value = [x for x in (coerce_value(v, key, desc) for v in value) if x is not None]
    else:
        value = coerce_value(value, key, desc)
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.5

    if value is not None and conf < config.CONFIDENCE_NULL_THRESHOLD:
        logger.info(f"Nulling {key}: confidence {conf:.2f} below threshold")
        value = None
    if value is None:
        default = absent_default(desc)
        if default is not None:
            return FieldResult(value=default, citation="", confidence=0.7)
        return FieldResult(value=None, citation="", confidence=round(max(conf, 0.5), 2))

    # Trust check: a citation we cannot find in the document lowers confidence
    if citation and squashed_text and not citation_supported(citation, squashed_text):
        conf = min(conf, 0.6)
    return FieldResult(
        value=value,
        citation=citation[:500],
        confidence=round(conf, 2),
        page_numbers=pages_in(citation) if doc_type == "pdf" else None,
        source_locations=source_locations,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Table documents: chunking
# ═════════════════════════════════════════════════════════════════════════════
_PAGE_LINE_RE = re.compile(r"^\[Page \d+\]")


@dataclass
class Chunk:
    label: str
    lines: list[str]
    split_ok: list[bool]          # split_ok[i]: a chunk may start at lines[i]
    context: str
    header: str = ""              # grid header line rendered above the lines
    sheet: Optional[dict] = None  # grid dict for spreadsheet chunks
    records: int = 0

    def body(self) -> str:
        return (self.header + "\n" if self.header else "") + "\n".join(self.lines)

    def split(self) -> Optional[tuple["Chunk", "Chunk"]]:
        """Split near the middle at a permitted boundary."""
        cands = [i for i in range(1, len(self.lines)) if self.split_ok[i]]
        if not cands:
            return None
        mid = len(self.lines) / 2
        i = min(cands, key=lambda c: abs(c - mid))
        b_lines, b_ok = self.lines[i:], self.split_ok[i:]
        if self.sheet is None and not _PAGE_LINE_RE.match(b_lines[0]):
            # Keep the second half's citations page-accurate
            marker = next((ln for ln in reversed(self.lines[:i]) if _PAGE_LINE_RE.match(ln)), None)
            if marker:
                b_lines = [_PAGE_LINE_RE.match(marker).group(0) + " (continued)"] + b_lines
                b_ok = [True] + b_ok
        a = Chunk(self.label + " (a)", self.lines[:i], self.split_ok[:i], self.context, self.header, self.sheet)
        tail = "\n".join(self.lines[max(0, i - GRID_PRECEDING_ROWS):i])
        b_ctx = (self.context + "\n\n" if self.context else "") + "... (lines just above this part)\n" + tail
        b = Chunk(self.label + " (b)", b_lines, b_ok, b_ctx, self.header, self.sheet)
        return a, b


def _grid_record_starts(grid: dict) -> set[int]:
    """Rows whose first used column is filled: in block layouts (unit line + charge lines) these
    start a record; in flat tables every row qualifies."""
    if not grid["columns"]:
        return set()
    first = grid["columns"][0]
    return {r for r, cells in grid["rows"] if cells and cells[0][0] == first}


def _is_label_row(cells: list[tuple[str, str]]) -> bool:
    """A header-like row: no numeric values beyond (possibly) a leading code column."""
    return bool(cells) and all(try_parse_number(t) is None for _, t in cells[1:])


def build_grid_chunks(doc: dict) -> list[Chunk]:
    sheets = doc["sheets"]
    chunks: list[Chunk] = []
    for grid in sheets:
        cols = grid["columns"]
        header = render_grid_header(grid["sheet_name"], cols)
        starts = _grid_record_starts(grid)
        rendered = [(r, render_grid_row(cols, r, cells)) for r, cells in grid["rows"]]
        head_rows = "\n".join(line for _, line in rendered[:GRID_CONTEXT_ROWS])
        other_ctx = "\n\n".join(
            s["text"] for s in sheets if s is not grid and len(s["text"]) <= SMALL_SHEET_CONTEXT_CHARS
        )

        label_only = [i for i, (_, cells) in enumerate(grid["rows"]) if _is_label_row(cells)]

        cur_lines: list[str] = []
        cur_ok: list[bool] = []
        size = records = 0
        first_row = None
        first_idx = 0

        def flush(last_row):
            nonlocal cur_lines, cur_ok, size, records, first_row
            if cur_lines:
                is_first = first_idx == 0
                ctx_parts = [] if is_first else [f"{header}\n{head_rows}"]
                if not is_first:
                    # Section headers that apply to the first rows of this part sit above it:
                    # include the nearest label-only (header) rows and the rows just above.
                    near = max(GRID_CONTEXT_ROWS, first_idx - GRID_PRECEDING_ROWS)
                    labels = [i for i in label_only if GRID_CONTEXT_ROWS <= i < near][-GRID_PRECEDING_LABELS:]
                    above = [rendered[i][1] for i in labels] + [ln for _, ln in rendered[near:first_idx]]
                    if above:
                        ctx_parts.append("... (rows above this part)\n" + "\n".join(above))
                if other_ctx:
                    ctx_parts.append(other_ctx)
                chunks.append(Chunk(
                    label=f"sheet '{grid['sheet_name']}' rows {first_row}-{last_row}",
                    lines=cur_lines, split_ok=cur_ok, context="\n\n".join(ctx_parts),
                    header=header, sheet=grid, records=records,
                ))
            cur_lines, cur_ok, size, records, first_row = [], [], 0, 0, None

        prev_row = None
        for idx, (r, line) in enumerate(rendered):
            is_start = r in starts
            over = size + len(line) > config.TABLE_CHUNK_MAX_CHARS or records >= config.TABLE_CHUNK_MAX_RECORDS
            hard_over = size + len(line) > 2 * config.TABLE_CHUNK_MAX_CHARS
            if cur_lines and ((over and is_start) or hard_over):
                flush(prev_row)
            if first_row is None:
                first_row = r
                first_idx = idx
            cur_lines.append(line)
            cur_ok.append(is_start)
            size += len(line) + 1
            records += int(is_start)
            prev_row = r
        flush(prev_row)
    return chunks


def build_text_chunks(doc: dict) -> list[Chunk]:
    """Page-aware chunks for PDF/DOCX tables. Each chunk gets the document head and the tail of
    the previous chunk as context (headers, section names that sit just above a page break)."""
    max_chars = config.TEXT_TABLE_CHUNK_MAX_CHARS
    lines: list[str] = []
    ok: list[bool] = []
    for p in doc["pages"]:
        lines.append(f"[Page {p['page']}]")
        ok.append(True)
        for ln in p["text"].split("\n"):
            if ln.strip():
                lines.append(ln)
                ok.append(True)

    head = doc["text"][:TEXT_HEAD_CONTEXT_CHARS]
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_ok: list[bool] = []
    size = 0
    current_page = 1
    chunk_first_page = 1
    prev_tail = ""
    for ln, can in zip(lines, ok):
        m = re.match(r"^\[Page (\d+)\]$", ln)
        if cur and size + len(ln) > max_chars:
            ctx = head if not chunks else head + "\n...\n[end of previous part]\n" + prev_tail
            # Lines that continue a page need the page marker repeated so citations stay correct
            chunks.append(Chunk(f"pages {chunk_first_page}-{current_page}", cur, cur_ok, ctx))
            prev_tail = "\n".join(cur)[-TEXT_TAIL_CONTEXT_CHARS:]
            cur, cur_ok, size = [], [], 0
            chunk_first_page = current_page
            if not m:
                cur.append(f"[Page {current_page}] (continued)")
                cur_ok.append(True)
        if m:
            current_page = int(m.group(1))
            if not cur:
                chunk_first_page = current_page
        cur.append(ln)
        cur_ok.append(can)
        size += len(ln) + 1
    if cur:
        ctx = head if not chunks else head + "\n...\n[end of previous part]\n" + prev_tail
        chunks.append(Chunk(f"pages {chunk_first_page}-{current_page}", cur, cur_ok, ctx))
    # A single chunk already contains the head; no need to repeat it
    if len(chunks) == 1:
        chunks[0].context = ""
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# Table documents: extraction
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class ChunkResult:
    order: float
    document: dict
    rows: list[dict]
    sheet: Optional[dict]


async def _run_chunk(schema, chunk: Chunk, doc_keys, row_keys, usage, order: float,
                     span: float = 1.0, depth: int = 0) -> tuple[list[ChunkResult], list[str]]:
    """Extract one chunk. On truncation or unusable output, split it and recurse."""
    system, user = build_table_chunk_prompt(
        schema, doc_keys, row_keys, chunk.body(), chunk.context, chunk.label, is_grid=chunk.sheet is not None,
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    problem = ""
    for attempt in range(2):
        resp = await chat_completion(
            model=config.TABLE_MODEL, messages=messages, usage_stats=usage,
            max_tokens=config.MAX_OUTPUT_TOKENS, response_format={"type": "json_object"},
            reasoning=config.TABLE_REASONING,
        )
        if resp.truncated:
            problem = "output truncated"
            break  # retrying the same chunk would truncate again: split instead
        try:
            parsed = parse_json_response(resp.content)
            rows = parsed.get("rows", parsed.get("entries", []))
            if not isinstance(rows, list):
                raise ValueError("'rows' is not a list")
            document = parsed.get("document") if isinstance(parsed.get("document"), dict) else {}
            return [ChunkResult(order, document, [r for r in rows if isinstance(r, dict)], chunk.sheet)], []
        except ValueError as e:
            problem = f"unusable output: {e}"
            logger.warning(f"Chunk {chunk.label}: {problem} (attempt {attempt + 1})")

    halves = chunk.split() if depth < MAX_SPLIT_DEPTH else None
    if halves:
        logger.info(f"Chunk {chunk.label}: {problem}; splitting in two")
        results = await asyncio.gather(
            *(_run_chunk(schema, h, doc_keys, row_keys, usage, order + i * span / 2, span / 2, depth + 1)
              for i, h in enumerate(halves))
        )
        out, warns = [], []
        for res, w in results:
            out += res
            warns += w
        return out, warns
    logger.error(f"Chunk {chunk.label} failed permanently: {problem}")
    return [], [f"{chunk.label}: {problem}"]


async def _extract_table(schema: ExtractionSchema, doc: dict, usage: UsageStats) -> ExtractionOutcome:
    pk = primary_key(schema)
    doc_keys = doc_level_keys(schema)
    row_keys = [k for k in schema.fields if k not in doc_keys]

    chunks = build_grid_chunks(doc) if doc.get("sheets") else build_text_chunks(doc)
    logger.info(f"Table extraction: {len(chunks)} chunk(s), doc-level fields={doc_keys}")

    # Order key: chunk i owns [i, i+1); split halves subdivide it, so document order is preserved
    gathered = await asyncio.gather(
        *(_run_chunk(schema, c, doc_keys, row_keys, usage, float(i)) for i, c in enumerate(chunks)),
        return_exceptions=True,
    )
    results: list[ChunkResult] = []
    warnings: list[str] = []
    for c, g in zip(chunks, gathered):
        if isinstance(g, Exception):
            warnings.append(f"{c.label}: {g}")
            logger.error(f"Chunk {c.label} raised: {g}")
            continue
        res, w = g
        results += res
        warnings += w
    if not results:
        raise RuntimeError("every chunk failed: " + "; ".join(warnings[:5]))
    results.sort(key=lambda r: r.order)

    # Recall check: find identifier-shaped lines the first pass skipped and re-check only those
    try:
        recheck = _build_recall_chunk(doc, schema, pk, results)
        if recheck is not None:
            res, w = await _run_chunk(schema, recheck, doc_keys, row_keys, usage, float(len(chunks)))
            for r in res:
                r.document = {}  # recheck context is partial; don't let it vote on header fields
            results += res
            warnings += w
    except Exception as e:  # noqa: BLE001 - the recall pass is best-effort
        logger.warning(f"Recall check failed: {e}")

    squashed = _squash(doc["text"])
    doc_fields = _vote_document_fields(schema, doc_keys, results, doc, squashed)
    entries = _merge_rows(schema, pk, row_keys, results, doc, squashed)
    for e in entries:
        for k in doc_keys:
            e[k] = doc_fields[k]
    # Return fields in schema order
    entries = [{k: e[k] for k in schema.fields} for e in entries]
    return ExtractionOutcome(entries, partial=bool(warnings), warnings=warnings)


# ── Recall check ─────────────────────────────────────────────────────────────
RECALL_MAX_CANDIDATES = 60
_ANY_NUMBER_RE = re.compile(r"(?<![\w.])[-($]*\d[\d,]*(\.\d+)?%?\)?(?![\w])")


def identifier_shapes(values: list[str]) -> list[re.Pattern]:
    """Regexes describing how the extracted identifiers look (digit runs, letter runs, punctuation).
    Returns the shapes that together cover >= 90% of values, or [] if they are too heterogeneous."""
    def shape(v: str) -> str:
        out, prev = [], None
        for ch in v.strip():
            kind = "d" if ch.isdigit() else "a" if ch.isalpha() else "s" if ch.isspace() else ch
            if kind == prev and kind in "das":
                continue
            out.append({"d": r"\d+", "a": r"[A-Za-z]+", "s": r"\s+"}.get(kind, re.escape(ch)))
            prev = kind
        return "".join(out)

    shapes = Counter(shape(v) for v in values if v)
    total = sum(shapes.values())
    chosen, covered = [], 0
    for shp, n in shapes.most_common(3):
        chosen.append(shp)
        covered += n
        if covered >= 0.9 * total:
            break
    if covered < 0.9 * total or any(shp in (r"\d+", r"[A-Za-z]+") for shp in chosen):
        return []  # bare numbers/words would match far too much
    return [re.compile(shp) for shp in chosen]


def _build_recall_chunk(doc: dict, schema, pk: str, results: list[ChunkResult]) -> Optional[Chunk]:
    desc = schema.fields[pk].description
    found = [coerce_value(r.get(pk), pk, desc) for res in results for r in res.rows]
    found = [str(v) for v in found if v is not None]
    if len(found) < 3:
        return None
    shapes = identifier_shapes(found)
    if not shapes:
        return None
    have = {norm_key(v) for v in found}

    def is_candidate(token: str) -> bool:
        return any(p.fullmatch(token) for p in shapes) and norm_key(token) not in have

    if doc.get("sheets"):
        lines, missing, ctx = [], [], []
        for grid in doc["sheets"]:
            cols = grid["columns"]
            for idx, (r, cells) in enumerate(grid["rows"]):
                for col, text in cells:
                    if is_candidate(text) and any(try_parse_number(t) is not None for c, t in cells if c != col):
                        missing.append(text)
                        block = _grid_block(grid, r) or [(r, cells)]
                        before = grid["rows"][max(0, idx - GRID_PRECEDING_ROWS):idx]
                        ctx += [render_grid_row(cols, rr, cc) for rr, cc in before]
                        lines += [render_grid_row(cols, rr, cc) for rr, cc in block]
                        break
                if len(missing) >= RECALL_MAX_CANDIDATES:
                    break
            if missing:
                header = render_grid_header(grid["sheet_name"], cols)
                head = "\n".join(render_grid_row(cols, rr, cc) for rr, cc in grid["rows"][:GRID_CONTEXT_ROWS])
                break  # one sheet per recheck keeps cell grounding simple
        if not missing:
            return None
        context = f"{header}\n{head}\n... (lines just above the lines to re-check)\n" + "\n".join(dict.fromkeys(ctx))
        chunk = Chunk("", list(dict.fromkeys(lines)), [True] * len(dict.fromkeys(lines)), context, header, grid)
    else:
        lines, missing = [], []
        page_lines = []
        for p in doc["pages"]:
            for ln in p["text"].split("\n"):
                if ln.strip():
                    page_lines.append((p["page"], ln))
        for i, (page, ln) in enumerate(page_lines):
            m = re.match(r"^[|\s]*(\S+(?:\s-\s\S+)?)", ln)
            token = m.group(1) if m else ""
            if token and is_candidate(token) and _ANY_NUMBER_RE.search(ln[m.end():]):
                missing.append(token)
                lines.append(f"[Page {page}]")
                lines += [l for _, l in page_lines[max(0, i - 3):i + 2]]
                if len(missing) >= RECALL_MAX_CANDIDATES:
                    break
        if not missing:
            return None
        chunk = Chunk("", lines, [True] * len(lines), doc["text"][:TEXT_HEAD_CONTEXT_CHARS])

    missing = list(dict.fromkeys(missing))
    logger.info(f"Recall check: {len(missing)} identifier(s) found in the document but not extracted: {missing[:10]}")
    chunk.label = ("RE-CHECK of lines a first pass skipped. Consider ONLY these identifiers: "
                   + ", ".join(missing) + ". Return a row for each one that qualifies under the row "
                   "definition, and none for lines that do not qualify (e.g. headers or excluded totals). "
                   "Surrounding lines are shown for context only")
    return chunk


def _vote_document_fields(schema, doc_keys, results: list[ChunkResult], doc, squashed) -> dict[str, FieldResult]:
    out: dict[str, FieldResult] = {}
    for k in doc_keys:
        desc = schema.fields[k].description
        candidates = []
        for r in results:
            raw = r.document.get(k)
            if isinstance(raw, dict):
                v, cit = coerce_value(raw.get("value"), k, desc), str(raw.get("citation") or "")
            else:
                v, cit = coerce_value(raw, k, desc), ""
            if v is not None and v != []:
                candidates.append((norm_key(v) if not isinstance(v, list) else json.dumps(v), v, cit))
        if not candidates:
            out[k] = FieldResult(value=None, citation="", confidence=0.5)
            continue
        counts = Counter(c[0] for c in candidates)
        best_key, best_n = counts.most_common(1)[0]
        _, value, cit = next(c for c in candidates if c[0] == best_key)
        agreement = best_n / len(candidates)
        conf = 0.6 + 0.35 * agreement
        locs = None
        if doc.get("sheets"):
            hit = _find_in_sheets_head(value, doc["sheets"])
            if hit:
                cit, locs = hit
                conf = max(conf, 0.9)
        out[k] = _finalize_field(k, desc, value, cit, conf, squashed, doc["type"], locs)
    return out


def _find_in_sheets_head(value, sheets, max_rows: int = 25):
    for grid in sheets:
        for r, cells in grid["rows"][:max_rows]:
            for col, text in cells:
                if values_match(value, text) or (isinstance(value, str) and len(value) > 3
                                                 and _ws_fold(value) in _ws_fold(text)):
                    ref = f"{grid['sheet_name']}!{col}{r}"
                    return f"[{ref}] {text}", [ref]
    return None


def _ws_fold(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def _grid_block(grid: dict, row_num: int) -> list[tuple[int, list[tuple[str, str]]]]:
    """Rows belonging to the record starting at row_num (until the next record start)."""
    cache = grid.setdefault("_block_cache", {})
    if not cache:
        starts = _grid_record_starts(grid)
        cache["starts"] = starts
        cache["index"] = {r: i for i, (r, _) in enumerate(grid["rows"])}
    idx = cache["index"].get(row_num)
    if idx is None:
        return []
    block = [grid["rows"][idx]]
    for r, cells in grid["rows"][idx + 1:]:
        if r in cache["starts"]:
            break
        block.append((r, cells))
    return block


def _merge_rows(schema, pk, row_keys, results: list[ChunkResult], doc, squashed) -> list[dict[str, FieldResult]]:
    merged: dict[Any, dict[str, FieldResult]] = {}
    order: list[Any] = []
    for res in results:
        for raw in res.rows:
            pk_val = coerce_value(raw.get(pk), pk, schema.fields[pk].description)
            if pk_val is None or (isinstance(pk_val, str) and not pk_val.strip()):
                continue  # a row without its identifier cannot be scored or trusted
            entry = _build_row(schema, row_keys, raw, res.sheet, doc, squashed)
            rid = norm_key(pk_val)
            if rid in merged:
                # First occurrence in document order wins: a unit repeated later (e.g. in a
                # future-residents or summary section) is not the row the table defines.
                continue
            merged[rid] = entry
            order.append(rid)
    return [merged[r] for r in order]


def _build_row(schema, row_keys, raw: dict, grid: Optional[dict], doc, squashed) -> dict[str, FieldResult]:
    row_cite = str(raw.get("_cite") or "")
    block = []
    row_num = None
    if grid is not None:
        try:
            row_num = int(str(raw.get("_row", "")).lstrip("rR"))
        except ValueError:
            row_num = None
        if row_num is not None:
            block = _grid_block(grid, row_num)

    entry: dict[str, FieldResult] = {}
    for k in row_keys:
        desc = schema.fields[k].description
        value = raw.get(k)
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, list):
            value = [x for x in (coerce_value(v, k, desc) for v in value) if x is not None]
        else:
            value = coerce_value(value, k, desc)
        if value is None:
            default = absent_default(desc)
            entry[k] = FieldResult(value=default, citation="", confidence=0.7)
            continue

        if grid is not None:
            hit = None
            for r, cells in block:
                for col, text in cells:
                    if values_match(value, text):
                        hit = (f"{grid['sheet_name']}!{col}{r}", text)
                        break
                if hit:
                    break
            if hit:
                entry[k] = FieldResult(value=value, citation=f"[{hit[0]}] {hit[1]}", confidence=0.95,
                                       source_locations=[hit[0]])
            else:
                # Computed (e.g. a sum of charge lines) or not grounded: cite the record's rows
                if block:
                    ref = f"{grid['sheet_name']}!A{block[0][0]}:A{block[-1][0]}"
                    cite = f"[{ref}] computed from record rows {block[0][0]}-{block[-1][0]}"
                    entry[k] = FieldResult(value=value, citation=cite, confidence=0.8, source_locations=[ref])
                else:
                    entry[k] = FieldResult(value=value, citation="", confidence=0.6)
        else:
            conf = 0.85 if (not row_cite or citation_supported(row_cite, squashed)) else 0.6
            entry[k] = FieldResult(
                value=value, citation=row_cite[:300], confidence=conf,
                page_numbers=pages_in(row_cite) if doc["type"] == "pdf" else None,
            )
    return entry
