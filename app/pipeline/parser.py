"""Document parsers: convert PDF, DOCX, XLSX, CSV into structured text for LLM consumption."""
from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── CSV ──────────────────────────────────────────────────────────────────────
def parse_csv(file_path: str) -> dict:
    """Parse CSV into markdown table with cell references."""
    import pandas as pd

    df = pd.read_csv(file_path)
    lines = []
    lines.append(f"**Source: CSV file ({Path(file_path).name})**\n")

    # Build markdown table with cell references
    headers = list(df.columns)
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    cell_map = {}  # (row, col) -> cell ref
    for row_idx, row in df.iterrows():
        row_vals = []
        for col_idx, val in enumerate(row):
            cell_ref = f"R{row_idx+2}C{col_idx+1}"  # 1-based, row 1 is header
            col_letter = chr(65 + col_idx) if col_idx < 26 else f"C{col_idx+1}"
            cell_ref = f"{col_letter}{row_idx+2}"
            cell_map[(row_idx, col_idx)] = cell_ref
            row_vals.append(str(val) if pd.notna(val) else "")
        lines.append("| " + " | ".join(row_vals) + " |")

    return {
        "type": "csv",
        "text": "\n".join(lines),
        "pages": [{"page": 1, "text": "\n".join(lines)}],
        "num_rows": len(df),
        "headers": headers,
        "raw_data": df.to_dict(orient="records"),
    }


# ── XLSX ─────────────────────────────────────────────────────────────────────
def parse_xlsx(file_path: str) -> dict:
    """Parse all sheets of an XLSX file into markdown tables with cell references."""
    import openpyxl

    wb = openpyxl.load_workbook(file_path, data_only=True)
    all_text_parts = []
    sheets_data = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        if ws.max_row is None or ws.max_row == 0:
            continue

        lines = [f"\n**Sheet: {sheet_name}**\n"]

        rows_data = []
        for row_idx, row in enumerate(ws.iter_rows(values_only=False), start=1):
            row_vals = []
            for cell in row:
                val = cell.value
                if val is None:
                    row_vals.append("")
                else:
                    row_vals.append(str(val))
            rows_data.append(row_vals)

        if not rows_data:
            continue

        # Build markdown table
        max_cols = max(len(r) for r in rows_data) if rows_data else 0
        if max_cols == 0:
            continue

        # Pad rows
        for r in rows_data:
            while len(r) < max_cols:
                r.append("")

        # Header row
        lines.append("| " + " | ".join(rows_data[0]) + " |")
        lines.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")

        for row in rows_data[1:]:
            lines.append("| " + " | ".join(row) + " |")

        sheet_text = "\n".join(lines)
        all_text_parts.append(sheet_text)
        sheets_data.append({
            "sheet_name": sheet_name,
            "text": sheet_text,
            "num_rows": len(rows_data) - 1,  # exclude header
        })

    return {
        "type": "xlsx",
        "text": "\n".join(all_text_parts),
        "pages": [{"page": i + 1, "text": s["text"], "sheet_name": s["sheet_name"]}
                  for i, s in enumerate(sheets_data)],
        "sheets": sheets_data,
    }


# ── DOCX ─────────────────────────────────────────────────────────────────────
def parse_docx(file_path: str) -> dict:
    """Parse DOCX into structured text with paragraph numbers."""
    from docx import Document

    doc = Document(file_path)
    paragraphs = []
    full_text_parts = []

    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            paragraphs.append({"index": i, "text": text, "style": para.style.name if para.style else ""})
            full_text_parts.append(text)

    # Also extract tables
    table_texts = []
    for t_idx, table in enumerate(doc.tables):
        lines = [f"\n**Table {t_idx + 1}:**\n"]
        for row_idx, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            lines.append("| " + " | ".join(cells) + " |")
            if row_idx == 0:
                lines.append("| " + " | ".join("---" for _ in cells) + " |")
        table_text = "\n".join(lines)
        table_texts.append(table_text)
        full_text_parts.append(table_text)

    full_text = "\n\n".join(full_text_parts)

    return {
        "type": "docx",
        "text": full_text,
        "pages": [{"page": 1, "text": full_text}],  # DOCX doesn't have pages
        "paragraphs": paragraphs,
    }


# ── PDF ──────────────────────────────────────────────────────────────────────
def parse_pdf(file_path: str) -> dict:
    """Parse PDF pages into text. Detect scanned pages (no text layer)."""
    import pdfplumber

    pages = []
    scanned_pages = []
    all_text_parts = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []

            # Build page text
            page_text_parts = [text]

            # Add any tables as markdown
            for t_idx, table in enumerate(tables):
                if table:
                    lines = []
                    for row_idx, row in enumerate(table):
                        cells = [str(c).strip() if c else "" for c in row]
                        lines.append("| " + " | ".join(cells) + " |")
                        if row_idx == 0:
                            lines.append("| " + " | ".join("---" for _ in cells) + " |")
                    page_text_parts.append("\n".join(lines))

            page_text = "\n".join(page_text_parts)

            # Detect scanned page: very little text
            is_scanned = len(text.strip()) < 50 and len(tables) == 0
            if is_scanned:
                scanned_pages.append(i + 1)

            pages.append({
                "page": i + 1,
                "text": page_text,
                "is_scanned": is_scanned,
                "has_tables": len(tables) > 0,
            })
            all_text_parts.append(f"[Page {i + 1}]\n{page_text}")

    return {
        "type": "pdf",
        "text": "\n\n".join(all_text_parts),
        "pages": pages,
        "scanned_pages": scanned_pages,
        "total_pages": len(pages),
    }


def pdf_page_to_image_base64(file_path: str, page_number: int) -> Optional[str]:
    """Convert a single PDF page to a base64-encoded PNG image."""
    import base64
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(file_path, first_page=page_number, last_page=page_number, dpi=200)
        if images:
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to convert page {page_number} to image: {e}")
    return None


# ── Dispatcher ───────────────────────────────────────────────────────────────
def parse_document(file_path: str, filename: str) -> dict:
    """Parse any supported document type. Returns structured content dict."""
    ext = Path(filename).suffix.lower()
    logger.info(f"Parsing {filename} (type: {ext})")

    if ext == ".csv":
        return parse_csv(file_path)
    elif ext == ".xlsx":
        return parse_xlsx(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext == ".pdf":
        return parse_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

