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
    assert len(res["headers"]) > 0
    assert "text" in res
    assert len(res["text"]) > 100


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
