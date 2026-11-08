"""Evidence page selection: unit building, assembly, and every fallback to the full document."""
import json

import pytest

from app import config
from app.llm.client import LLMResponse, UsageStats
from app.models import ExtractionSchema, SchemaField
from app.pipeline import selector
from app.pipeline.extractor import parse_json_response

SCHEMA = ExtractionSchema(name="lease_abstraction", fields={
    "tenant_name": SchemaField(name="Tenant", description="Tenant legal name."),
    "base_rent": SchemaField(name="Rent", description="Monthly base rent, as a number."),
})


def _doc(n_pages=10, page_chars=4000):
    pages = [{"page": i, "text": f"page {i} " + "x" * page_chars} for i in range(1, n_pages + 1)]
    return {"type": "pdf", "pages": pages, "text": "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages)}


def _fake(content: str):
    calls = []

    async def fake(model, messages, usage_stats, **kw):
        calls.append(model)
        return LLMResponse(content, "stop", 10, 10, 0.0)
    return fake, calls


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setattr(config, "DOCUMENT_PAGE_SELECTION", True)
    monkeypatch.setattr(config, "SELECTION_MIN_CHARS", 1000)
    monkeypatch.setattr(config, "SELECTION_MAX_SHARE", 0.8)


def test_docx_is_split_into_blocks_on_paragraphs():
    text = "\n\n".join(f"para {i} " + "y" * 900 for i in range(20))
    units = selector.build_units({"pages": [{"page": 1, "text": text}], "text": text})
    assert len(units) > 3 and all(u.label.startswith("Block") for u in units)
    assert "\n\n".join(u.body for u in units) == text  # nothing lost or reordered


def test_assemble_keeps_order_and_neighbour_edges():
    units = selector.build_units(_doc(6, 2000))
    out = selector._assemble(units, {0, 3})
    assert out.index("[Page 1]") < out.index("[Page 4]")
    assert "[...end of Page 3]" in out and "[start of Page 5...]" in out
    assert "[Page 6]" not in out


async def test_selection_keeps_opening_and_picked_pages(monkeypatch):
    fake, calls = _fake(json.dumps({"pages": {"tenant_name": [1], "base_rent": [7]}}))
    monkeypatch.setattr(selector, "chat_completion", fake)
    text, info = await selector.select_text(SCHEMA, _doc(), UsageStats(), parse_json_response)
    assert calls == [config.SELECT_MODEL]
    assert "[Page 1]" in text and "[Page 2]" in text and "[Page 7]" in text
    assert "[Page 5]" not in text and info["selection"] == "3/10 pages"


@pytest.mark.parametrize("content", ["not json", json.dumps({"pages": {"tenant_name": list(range(1, 11))}})])
async def test_falls_back_to_full_text(monkeypatch, content):
    """Unusable answer, or a selection that is most of the document -> full text."""
    fake, _ = _fake(content)
    monkeypatch.setattr(selector, "chat_completion", fake)
    doc = _doc()
    text, info = await selector.select_text(SCHEMA, doc, UsageStats(), parse_json_response)
    assert text == doc["text"] and info["selection"] == "full"


async def test_short_documents_and_switch_off_skip_the_call(monkeypatch):
    fake, calls = _fake("{}")
    monkeypatch.setattr(selector, "chat_completion", fake)
    monkeypatch.setattr(config, "SELECTION_MIN_CHARS", 10**9)
    await selector.select_text(SCHEMA, _doc(), UsageStats(), parse_json_response)
    monkeypatch.setattr(config, "SELECTION_MIN_CHARS", 1000)
    monkeypatch.setattr(config, "DOCUMENT_PAGE_SELECTION", False)
    await selector.select_text(SCHEMA, _doc(), UsageStats(), parse_json_response)
    assert calls == []
