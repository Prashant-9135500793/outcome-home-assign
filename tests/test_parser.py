"""Tests for document parsers on sample documents."""
import os
import pytest
from app.pipeline.parser import parse_document, parse_csv, parse_xlsx, parse_docx, parse_pdf

SAMPLE_DOCS = "sample/sample-data-documents"


def test_parse_csv():
    path = os.path.join(SAMPLE_DOCS, "boxscore_001.csv")
    assert os.path.exists(path), f"Sample file {path} not found"
    res = parse_csv(path)
    assert res["type"] == "csv"
    assert res["num_rows"] > 0
    assert len(res["text"]) > 100
    # Title rows above the table are kept (pandas used to swallow "Box Score" as the header)
    rows = dict(res["sheets"][0]["rows"])
    assert rows[1] == [("A", "Box Score")]
    assert ("A", "Code") in rows[6]
    assert "| r7 | PR - P1 |" in res["text"]


def test_parse_xlsx_t12():
    path = os.path.join(SAMPLE_DOCS, "t12_001.xlsx")
    assert os.path.exists(path), f"Sample file {path} not found"
    res = parse_xlsx(path)
    assert res["type"] == "xlsx"
    assert len(res["sheets"]) > 0
    assert "text" in res
    assert len(res["text"]) > 100


def test_parse_xlsx_rentroll():
    path = os.path.join(SAMPLE_DOCS, "rentroll_001.xlsx")
    assert os.path.exists(path), f"Sample file {path} not found"
    res = parse_xlsx(path)
    assert res["type"] == "xlsx"
    assert len(res["sheets"]) > 0


def test_parse_docx():
    path = os.path.join(SAMPLE_DOCS, "lease_002.docx")
    assert os.path.exists(path), f"Sample file {path} not found"
    res = parse_docx(path)
    assert res["type"] == "docx"
    assert len(res["text"]) > 500
    assert len(res["paragraphs"]) > 0


def test_parse_pdf():
    path = os.path.join(SAMPLE_DOCS, "budget_001.pdf")
    assert os.path.exists(path), f"Sample file {path} not found"
    res = parse_pdf(path)
    assert res["type"] == "pdf"
    assert res["total_pages"] == 8
    assert len(res["text"]) > 500


def test_parse_document_dispatcher():
    csv_path = os.path.join(SAMPLE_DOCS, "boxscore_001.csv")
    res = parse_document(csv_path, "boxscore_001.csv")
    assert res["type"] == "csv"

    pdf_path = os.path.join(SAMPLE_DOCS, "lease_001.pdf")
    res_pdf = parse_document(pdf_path, "lease_001.pdf")
    assert res_pdf["type"] == "pdf"



def test_rentroll_render_is_compact_and_keeps_cell_positions():
    res = parse_xlsx(os.path.join(SAMPLE_DOCS, "rentroll_001.xlsx"))
    grid = res["sheets"][0]
    rows = dict(grid["rows"])
    # Unit line and its charge line keep their real columns
    assert ("A", "100-101") in rows[8]
    assert ("H", "Rent") in rows[10] and ("I", "2842") in rows[10]
    # Dates render as ISO, integers without ".0"
    assert ("L", "2022-12-03") in rows[8]
    # Empty trailing cells are trimmed: a charge line does not carry 30 empty columns
    line = [ln for ln in grid["text"].split("\n") if ln.startswith("| r10 |")][0]
    assert line.count("|") < 22


def test_xlsx_wide_merged_title_not_repeated():
    res = parse_xlsx(os.path.join(SAMPLE_DOCS, "t12_001.xlsx"))
    assert res["text"].count("12 Month Actual to Budget") == 1


def test_scanned_pdf_detected_and_renderable():
    from app.pipeline.parser import render_pdf_page_jpeg_base64
    path = os.path.join(SAMPLE_DOCS, "lease_003.pdf")
    res = parse_pdf(path)
    assert res["scanned_pages"] == list(range(1, res["total_pages"] + 1))
    img = render_pdf_page_jpeg_base64(path, 1)
    assert img and len(img) > 10_000


def test_concurrent_page_rendering_is_safe():
    """PDFium is not thread-safe; OCR renders pages from worker threads for several jobs at once."""
    import concurrent.futures as cf
    from app.pipeline.parser import render_pdf_page_jpeg_base64
    args = [(os.path.join(SAMPLE_DOCS, f), p) for f in ("lease_001.pdf", "lease_003.pdf") for p in range(1, 7)]
    with cf.ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda a: render_pdf_page_jpeg_base64(*a), args))
    assert all(out)
