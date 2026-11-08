"""Document parsers: turn PDF, DOCX, XLSX and CSV into text the LLM can read.

Every parser returns a dict with at least:
  type   "pdf" | "docx" | "xlsx" | "csv"
  text   the whole document rendered as text
  pages  list of {"page": n, "text": ...} (one per PDF page, one per sheet, one for DOCX)

Spreadsheets (XLSX and CSV) additionally return "sheets": a list of Grid-like dicts that
keep every non-empty cell with its row number and column letter. The extractor uses them to
chunk large tables on row boundaries and to ground citations to real cell references.
"""
from __future__ import annotations

import base64
import csv
import io
import logging
import math
import threading
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PDFIUM_LOCK = threading.Lock()

# Pages with less extractable text than this are treated as scanned and sent to OCR.
SCANNED_PAGE_MIN_CHARS = 50


# ── Cell formatting ──────────────────────────────────────────────────────────
def col_letter(idx: int) -> str:
    """0-based column index -> spreadsheet letter (0 -> A, 26 -> AA)."""
    s = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def format_cell(val: Any) -> str:
    """Render a cell value the way a person reads it: no '.0' floats, ISO dates."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, datetime):
        if val.time() == time(0, 0):
            return val.date().isoformat()
        return val.isoformat(sep=" ")
    if isinstance(val, date):
        return val.isoformat()
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return ""
        if val == int(val) and abs(val) < 1e15:
            return str(int(val))
        return repr(round(val, 6))
    return str(val).replace("\n", " ").replace("|", "/").strip()


# ── Grid (shared by XLSX and CSV) ────────────────────────────────────────────
def build_grid(name: str, raw_rows: list[list[Any]]) -> dict:
    """Build a sheet grid from raw rows (row 1 first).

    Returns {"sheet_name", "rows": [(row_num, [(col_letter, text), ...])], "columns": [...],
             "text", "num_rows"} where only non-empty rows/cells are kept and columns that are
    empty in every row are dropped from the rendered text (positions are preserved, so the
    letters still match the real spreadsheet).
    """
    rows: list[tuple[int, list[tuple[str, str]]]] = []
    used_cols: set[int] = set()
    for r_idx, raw in enumerate(raw_rows, start=1):
        cells = []
        for c_idx, v in enumerate(raw):
            s = format_cell(v)
            if s:
                cells.append((c_idx, s))
                used_cols.add(c_idx)
        if cells:
            rows.append((r_idx, cells))

    columns = sorted(used_cols)
    col_letters = [col_letter(c) for c in columns]
    grid_rows = [(r, [(col_letter(c), s) for c, s in cells]) for r, cells in rows]

    return {
        "sheet_name": name,
        "rows": grid_rows,
        "columns": col_letters,
        "text": render_grid(name, col_letters, grid_rows),
        "num_rows": len(grid_rows),
    }


def render_grid_header(name: str, col_letters: list[str]) -> str:
    return f"### Sheet: {name}\n| row | " + " | ".join(col_letters) + " |"


def render_grid_row(col_letters: list[str], row_num: int, cells: list[tuple[str, str]]) -> str:
    """One table line per row; trailing empty cells are trimmed to keep sparse sheets small."""
    by_col = dict(cells)
    last = max((i for i, c in enumerate(col_letters) if c in by_col), default=-1)
    return f"| r{row_num} | " + " | ".join(by_col.get(c, "") for c in col_letters[: last + 1]) + " |"


def render_grid(name: str, col_letters: list[str], rows: list) -> str:
    lines = [render_grid_header(name, col_letters)]
    lines += [render_grid_row(col_letters, r, cells) for r, cells in rows]
    return "\n".join(lines)


# ── CSV ──────────────────────────────────────────────────────────────────────
def parse_csv(file_path: str) -> dict:
    """Parse CSV as a raw grid (no header inference: many reports have title rows on top)."""
    raw = Path(file_path).read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            content = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialect = csv.Sniffer().sniff(content[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    raw_rows = list(csv.reader(io.StringIO(content), dialect))

    name = Path(file_path).stem
    grid = build_grid(name, raw_rows)
    return {
        "type": "csv",
        "text": grid["text"],
        "pages": [{"page": 1, "text": grid["text"], "sheet_name": name}],
        "sheets": [grid],
        "num_rows": grid["num_rows"],
    }


# ── XLSX ─────────────────────────────────────────────────────────────────────
def parse_xlsx(file_path: str) -> dict:
    """Parse every sheet of an XLSX into a grid. Merged cells take their top-left value."""
    import openpyxl

    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheets = []
    for ws in wb.worksheets:
        raw_rows = [list(r) for r in ws.iter_rows(values_only=True)]
        # Fill merged ranges so a merged label is visible on every row it spans. Fill across
        # columns only for narrow merges (grouped headers); wide merges are titles and
        # repeating them across the sheet only adds noise.
        for rng in ws.merged_cells.ranges:
            tl = ws.cell(rng.min_row, rng.min_col).value
            if tl is None:
                continue
            max_col = rng.max_col if rng.max_col - rng.min_col < 3 else rng.min_col
            for r in range(rng.min_row, rng.max_row + 1):
                for c in range(rng.min_col, max_col + 1):
                    if r - 1 < len(raw_rows):
                        row = raw_rows[r - 1]
                        while len(row) < c:
                            row.append(None)
                        if row[c - 1] is None:
                            row[c - 1] = tl
        grid = build_grid(ws.title, raw_rows)
        if grid["num_rows"]:
            sheets.append(grid)

    return {
        "type": "xlsx",
        "text": "\n\n".join(s["text"] for s in sheets),
        "pages": [{"page": i + 1, "text": s["text"], "sheet_name": s["sheet_name"]}
                  for i, s in enumerate(sheets)],
        "sheets": sheets,
    }


# ── DOCX ─────────────────────────────────────────────────────────────────────
def _iter_docx_blocks(doc):
    """Yield paragraphs and tables in document order."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield Paragraph(child, doc)
        elif tag == "tbl":
            yield Table(child, doc)


def parse_docx(file_path: str) -> dict:
    """Parse DOCX into text, keeping tables in place and rendered as markdown."""
    from docx import Document
    from docx.table import Table

    doc = Document(file_path)
    parts: list[str] = []
    paragraphs = []
    for i, block in enumerate(_iter_docx_blocks(doc)):
        if isinstance(block, Table):
            lines = []
            for r_idx, row in enumerate(block.rows):
                cells = []
                prev = None
                for cell in row.cells:
                    # python-docx repeats merged cells; collapse consecutive duplicates
                    if prev is not None and cell._tc is prev:
                        continue
                    prev = cell._tc
                    cells.append(cell.text.strip().replace("\n", " ").replace("|", "/"))
                lines.append("| " + " | ".join(cells) + " |")
                if r_idx == 0:
                    lines.append("| " + " | ".join("---" for _ in cells) + " |")
            if lines:
                parts.append("\n".join(lines))
        else:
            text = block.text.strip()
            if text:
                paragraphs.append({"index": i, "text": text, "style": block.style.name if block.style else ""})
                parts.append(text)

    # Headers/footers often carry the property name or document date
    hf = []
    for section in doc.sections:
        for p in list(section.header.paragraphs) + list(section.footer.paragraphs):
            t = p.text.strip()
            if t and t not in hf:
                hf.append(t)
    if hf:
        parts.insert(0, "[Header/Footer] " + " / ".join(hf))

    full_text = "\n\n".join(parts)
    return {
        "type": "docx",
        "text": full_text,
        "pages": [{"page": 1, "text": full_text}],  # DOCX has no fixed pages
        "paragraphs": paragraphs,
    }


# ── PDF ──────────────────────────────────────────────────────────────────────
def parse_pdf(file_path: str) -> dict:
    """Extract text per page with pdfplumber; flag pages with no text layer as scanned."""
    import pdfplumber

    pages = []
    scanned_pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:  # noqa: BLE001 - a broken page should not kill the document
                logger.warning(f"Text extraction failed on page {i + 1}: {e}")
                text = ""
            try:
                tables = page.extract_tables() or []
            except Exception:  # noqa: BLE001
                tables = []

            parts = [text]
            for table in tables:
                if not table:
                    continue
                lines = []
                for r_idx, row in enumerate(table):
                    cells = [str(c).replace("\n", " ").strip() if c else "" for c in row]
                    lines.append("| " + " | ".join(cells) + " |")
                    if r_idx == 0:
                        lines.append("| " + " | ".join("---" for _ in cells) + " |")
                parts.append("[Table]\n" + "\n".join(lines))

            is_scanned = len(text.strip()) < SCANNED_PAGE_MIN_CHARS
            if is_scanned:
                scanned_pages.append(i + 1)
            pages.append({
                "page": i + 1,
                "text": "\n".join(p for p in parts if p),
                "is_scanned": is_scanned,
                "has_tables": bool(tables),
            })

    return {
        "type": "pdf",
        "text": join_pages(pages),
        "pages": pages,
        "scanned_pages": scanned_pages,
        "total_pages": len(pages),
    }


def join_pages(pages: list[dict]) -> str:
    return "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages)


def render_pdf_page_jpeg_base64(file_path: str, page_number: int, max_side: int = 1700,
                                quality: int = 80) -> Optional[str]:
    """Render one PDF page (1-based) to a base64 JPEG using pdfium (no system deps)."""
    try:
        import pypdfium2 as pdfium

        # PDFium is not thread-safe: concurrent use corrupts state and can crash the process.
        # Rendering is ~0.1 s/page, negligible next to the vision call, so serialise it.
        with _PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(file_path)
            try:
                page = pdf[page_number - 1]
                w, h = page.get_size()  # points (1/72 inch)
                scale = min(max_side / max(w, h), 3.0)
                image = page.render(scale=scale).to_pil().convert("RGB")
            finally:
                pdf.close()
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to render page {page_number} of {file_path}: {e}")
        return None


# ── Dispatcher ───────────────────────────────────────────────────────────────
def parse_document(file_path: str, filename: str) -> dict:
    """Parse any supported document type. Returns a structured content dict."""
    ext = Path(filename).suffix.lower()
    logger.info(f"Parsing {filename} (type: {ext})")
    if ext == ".csv":
        return parse_csv(file_path)
    if ext == ".xlsx":
        return parse_xlsx(file_path)
    if ext == ".docx":
        return parse_docx(file_path)
    if ext == ".pdf":
        return parse_pdf(file_path)
    raise ValueError(f"Unsupported file type: {ext}")
