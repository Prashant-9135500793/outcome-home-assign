"""Extraction pipeline tests with a fake LLM: chunking, truncation recovery, merging, cleaning."""
import json
import os
import re

import pytest

from app import config
from app.llm.client import LLMResponse, UsageStats
from app.models import ExtractionSchema, SchemaField
from app.pipeline import extractor
from app.pipeline.extractor import (
    build_grid_chunks,
    build_text_chunks,
    coerce_value,
    extract_document,
    parse_json_response,
    values_match,
)
from app.pipeline.parser import parse_xlsx
from app.pipeline.prompts import doc_level_keys

SAMPLE_DOCS = "sample/sample-data-documents"


def _schema(fields: dict, name="t") -> ExtractionSchema:
    return ExtractionSchema(name=name, fields={k: SchemaField(**v) for k, v in fields.items()})


TABLE_SCHEMA = _schema({
    "property_name": {"name": "Property", "description": "Property name. Same for every row."},
    "code": {"name": "Code", "description": "Unit type code.", "primary": True},
    "units": {"name": "Units", "description": "Number of units."},
})


# ── JSON parsing ─────────────────────────────────────────────────────────────
def test_parse_json_handles_fences_and_prose():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Here you go: {"a": {"b": "x}"}} trailing') == {"a": {"b": "x}"}}
    with pytest.raises(ValueError):
        parse_json_response("no json here")


# ── Value cleaning ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("value,key,desc,expected", [
    ("$1,425.50", "avg_rent", "Average rent as a number", 1425.5),
    ("(1,234)", "variance", "Dollar variance", -1234),
    ("89.47%", "percent_occupied", "Value of the '% Occ' column exactly as printed, e.g. 89.47.", 89.47),
    ("6101-0000", "gl_code", "General ledger account code, exactly as shown", "6101-0000"),
    ("02/09/2025", "date", "The end date of the period, in YYYY-MM-DD.", "2025-02-09"),
    ("75001", "zip_code", "ZIP code as a string.", "75001"),
    ("N/A", "late_fee", "The late charge", None),
    (1205.0, "unit_square_feet", "Square footage", 1205),
    ("False", "early_termination_right", "true if the tenant may terminate early, false otherwise.", False),
    ("True", "letter_of_credit_required", "true if the lease requires a letter of credit, false otherwise.", True),
])
def test_coerce_value(value, key, desc, expected):
    assert coerce_value(value, key, desc) == expected


def test_values_match_numbers_dates_text():
    assert values_match(2842, "2842")
    assert values_match(2842.0, "$2,842.00")
    assert values_match("2022-12-03", "12/03/2022")
    assert values_match("Occupied No Notice", "occupied  no notice")
    assert not values_match(2842, "2843")


def test_doc_level_detection_from_descriptions():
    assert doc_level_keys(TABLE_SCHEMA) == ["property_name"]


# ── Chunking ─────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not os.path.isdir(SAMPLE_DOCS), reason="sample documents not present")
def test_rentroll_chunks_split_only_on_unit_boundaries(monkeypatch):
    monkeypatch.setattr(config, "TABLE_CHUNK_MAX_RECORDS", 40)
    doc = parse_xlsx(os.path.join(SAMPLE_DOCS, "rentroll_001.xlsx"))
    chunks = build_grid_chunks(doc)
    main = [c for c in chunks if c.sheet["sheet_name"] == doc["sheets"][0]["sheet_name"]]
    assert len(main) > 5
    for c in main[1:]:
        # Every chunk after the first starts on a unit line (column A filled) ...
        assert re.match(r"^\| r\d+ \| \S", c.lines[0]), c.lines[0]
        # ... and carries the report header as context
        assert "Willow Creek Commons" in c.context and "Bldg-Unit" in c.context
    assert all(c.records <= 40 for c in main)
    # No row is lost or duplicated across chunks
    all_lines = [ln for c in main for ln in c.lines]
    assert len(all_lines) == len(doc["sheets"][0]["rows"])


def test_text_chunks_keep_page_markers():
    pages = [{"page": i, "text": "\n".join(f"line {i}-{j} " + "x" * 80 for j in range(60))} for i in (1, 2, 3)]
    doc = {"type": "pdf", "pages": pages, "text": extractor.join_pages(pages)}
    chunks = build_text_chunks(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert c.lines[0].startswith("[Page ")
    a, b = chunks[0].split()
    assert b.lines[0].startswith("[Page ")


# ── End-to-end table extraction with a fake model ────────────────────────────
class FakeLLM:
    """Answers table-chunk prompts by reading rows back out of the prompt text.
    Truncates any chunk containing more than `truncate_over` data rows, to exercise splitting."""

    def __init__(self, truncate_over=None):
        self.calls = 0
        self.truncate_over = truncate_over

    async def __call__(self, model, messages, usage_stats, **kw):
        self.calls += 1
        usage_stats.record(model, 100, 50, 0.0001)
        user = messages[-1]["content"]
        part = user.split("<part", 1)[1]
        rows = re.findall(r"\| r(\d+) \| (PR - \w+) \| [^|]* \| (\d+) \|", part)
        if self.truncate_over and len(rows) > self.truncate_over:
            return LLMResponse('{"rows": [', "length", 100, 50, 0.0)
        body = {
            "document": {"property_name": {"value": "Pearl at Victory Ridge", "citation": "[r2]"}},
            "rows": [{"_row": int(r), "code": c, "units": int(u)} for r, c, u in rows],
        }
        return LLMResponse(json.dumps(body), "stop", 100, 50, 0.0)


def _write_csv(tmp_path, n_rows):
    lines = ["Box Score,,", "Pearl at Victory Ridge,,", "Code,Name,Units"]
    lines += [f"PR - U{i},Unit {i},{i + 1}" for i in range(n_rows)]
    p = tmp_path / "bs.csv"
    p.write_text("\n".join(lines))
    return str(p)


async def test_table_extraction_end_to_end(tmp_path, monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(extractor, "chat_completion", fake)
    path = _write_csv(tmp_path, 10)
    out = await extract_document(path, "bs.csv", TABLE_SCHEMA, UsageStats())
    assert not out.partial
    assert [e["code"].value for e in out.entries] == [f"PR - U{i}" for i in range(10)]
    assert all(e["property_name"].value == "Pearl at Victory Ridge" for e in out.entries)
    # Citations are grounded to the real cell
    first = out.entries[0]["units"]
    assert first.value == 1 and first.source_locations == ["bs!C4"] and first.confidence >= 0.9


async def test_truncated_chunk_is_split_and_recovered(tmp_path, monkeypatch):
    fake = FakeLLM(truncate_over=6)
    monkeypatch.setattr(extractor, "chat_completion", fake)
    path = _write_csv(tmp_path, 20)
    out = await extract_document(path, "bs.csv", TABLE_SCHEMA, UsageStats())
    assert not out.partial
    # All rows recovered, in document order, no duplicates
    assert [e["code"].value for e in out.entries] == [f"PR - U{i}" for i in range(20)]
    assert fake.calls > 1


async def test_single_entry_low_confidence_is_nulled(tmp_path, monkeypatch):
    schema = _schema({
        "tenant_name": {"name": "Tenant", "description": "Tenant legal name"},
        "parking_spaces": {"name": "Parking", "description": "Number of parking spaces"},
    })

    async def fake(model, messages, usage_stats, **kw):
        usage_stats.record(model, 10, 10, 0.0)
        return LLMResponse(json.dumps({"fields": {
            "tenant_name": {"value": "Acme Corp.", "citation": "Acme Corp., a Delaware corporation", "confidence": 0.95},
            "parking_spaces": {"value": "10", "citation": "", "confidence": 0.1},
        }}), "stop", 10, 10, 0.0)

    monkeypatch.setattr(extractor, "chat_completion", fake)
    p = tmp_path / "lease.csv"
    p.write_text("This lease is between Landlord LLC and Acme Corp., a Delaware corporation\n")
    out = await extract_document(str(p), "lease.csv", schema, UsageStats())
    e = out.entries[0]
    assert e["tenant_name"].value == "Acme Corp." and e["tenant_name"].confidence == 0.95
    assert e["parking_spaces"].value is None and e["parking_spaces"].citation == ""


def test_null_field_keeps_value_key_in_output():
    from app.models import FieldResult
    assert FieldResult(value=None, citation="", confidence=0.5).to_output() == \
        {"value": None, "citation": "", "confidence": 0.5}
    out = FieldResult(value=3, citation="[S!A1] 3", confidence=0.9, source_locations=["S!A1"]).to_output()
    assert out["source_locations"] == ["S!A1"] and "page_numbers" not in out


def test_numeric_looking_doc_value_grounds_without_crash():
    sheets = [{"sheet_name": "R", "rows": [(1, [("A", "Property =  417 .417")])]}]
    hit = extractor._find_in_sheets_head("417 .417", sheets)
    assert hit and hit[1] == ["R!A1"]


def test_identifier_shapes():
    shapes = extractor.identifier_shapes(["6101-0000", "6102-0000", "8080-0000"])
    assert shapes and shapes[0].fullmatch("7010-0001") and not shapes[0].fullmatch("2025")
    assert extractor.identifier_shapes(["1", "2", "3"]) == []  # bare numbers are too ambiguous


async def test_recall_check_recovers_dropped_row(tmp_path, monkeypatch):
    """First pass drops a row; the recall pass finds it by identifier shape and re-checks it."""
    calls = {"n": 0}

    async def fake(model, messages, usage_stats, **kw):
        calls["n"] += 1
        usage_stats.record(model, 10, 10, 0.0)
        user = messages[-1]["content"]
        part = user.split("<part", 1)[1]
        rows = re.findall(r"\| r(\d+) \| (PR - \w+) \| [^|]* \| (\d+) \|", part)
        if calls["n"] == 1:
            rows = [r for r in rows if r[1] != "PR - U3"]  # simulate the model skipping a row
        body = {"document": {}, "rows": [{"_row": int(r), "code": c, "units": int(u)} for r, c, u in rows]}
        return LLMResponse(json.dumps(body), "stop", 10, 10, 0.0)

    monkeypatch.setattr(extractor, "chat_completion", fake)
    out = await extract_document(_write_csv(tmp_path, 6), "bs.csv", TABLE_SCHEMA, UsageStats())
    assert sorted(e["code"].value for e in out.entries) == sorted(f"PR - U{i}" for i in range(6))
    assert calls["n"] == 2


def test_duplicate_identifier_keeps_first_occurrence():
    from app.pipeline.extractor import ChunkResult, _merge_rows
    schema = TABLE_SCHEMA
    results = [ChunkResult(0.0, {}, [{"code": "U1", "units": None}], None),
               ChunkResult(1.0, {}, [{"code": "U1", "units": 5}], None)]
    doc = {"type": "csv", "text": ""}
    rows = _merge_rows(schema, "code", ["code", "units"], results, doc, "")
    assert len(rows) == 1 and rows[0]["units"].value is None


@pytest.mark.parametrize("desc,expected", [
    ("Security deposit held for the unit, as a number. 0 if none.", 0),
    ("The scheduled monthly charge ... 0 if the unit has no rent charge line.", 0),
    ("true if the lease requires a letter of credit as security, false otherwise.", False),
    ("List of the brokers named in the lease. Empty list if none are named.", []),
    ("Outstanding resident balance for the unit, as a number.", None),
    ("Full name of any guarantor of the lease. null if there is no guaranty.", None),
])
def test_absent_default_follows_description(desc, expected):
    assert extractor.absent_default(desc) == expected
