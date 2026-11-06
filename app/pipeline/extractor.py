"""Core extraction logic: parses documents and calls LLMs to extract fields."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from app.config import (
    DEFAULT_MODEL,
    REASONING_MODEL,
    VISION_MODEL,
    CONFIDENCE_NULL_THRESHOLD,
)
from app.llm.client import UsageStats, chat_completion
from app.models import ExtractionSchema, FieldResult
from app.pipeline.parser import parse_document, pdf_page_to_image_base64
from app.pipeline.prompts import (
    build_extraction_prompt,
    build_table_extraction_prompt,
    build_chunked_extraction_prompt,
)

logger = logging.getLogger(__name__)

# Max text length to send in a single call (chars). Beyond this, chunk.
MAX_SINGLE_CALL_CHARS = 80_000
# Max tokens for output
MAX_OUTPUT_TOKENS = 16384


def _is_table_schema(schema: ExtractionSchema) -> bool:
    """Check if the schema describes a table (has a primary field)."""
    return any(f.primary for f in schema.fields.values())


def _pick_model(schema: ExtractionSchema, doc_content: dict) -> str:
    """Choose the right model based on document type and complexity."""
    doc_type = doc_content["type"]
    schema_name = schema.name.lower()

    # CSV and simple XLSX: cheap model is enough
    if doc_type in ("csv",):
        return DEFAULT_MODEL

    # Spreadsheets: usually structured, cheap model works
    if doc_type == "xlsx":
        return DEFAULT_MODEL

    # DOCX: moderate complexity
    if doc_type == "docx":
        # Leases in DOCX need reasoning
        if "lease" in schema_name:
            return REASONING_MODEL
        return DEFAULT_MODEL

    # PDF
    if doc_type == "pdf":
        # Check if we have scanned pages
        scanned = doc_content.get("scanned_pages", [])
        if scanned:
            return VISION_MODEL

        # Complex schemas need reasoning
        if schema_name in ("lease_abstraction", "offering_memorandum"):
            return REASONING_MODEL

        # Budget and other multi-page table PDFs
        total_pages = doc_content.get("total_pages", 1)
        if total_pages > 5:
            return REASONING_MODEL

        return DEFAULT_MODEL

    return DEFAULT_MODEL


def _parse_json_response(text: str) -> dict:
    """Robustly parse JSON from LLM response, handling markdown blocks etc."""
    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        # Remove first and last lines if they're code fences
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    # Look for the outermost { ... }
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    # Find matching closing brace
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start:i + 1])
                except json.JSONDecodeError:
                    break

    raise ValueError(f"Could not parse JSON from response: {text[:500]}")


def _coerce_value(value: Any, field_key: str, field_desc: str) -> Any:
    """Coerce extracted values to proper types."""
    if value is None:
        return None

    desc_lower = field_desc.lower()

    # Handle string numbers that should be numeric
    if isinstance(value, str):
        # Check if it looks like a number
        cleaned = value.replace("$", "").replace(",", "").replace("%", "").strip()
        # Parentheses = negative
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]

        # Try numeric conversion if description suggests a number
        numeric_hints = ["number", "amount", "rent", "cost", "price", "sf", "sqft",
                         "square", "units", "count", "months", "rate", "ratio",
                         "variance", "total", "budget", "deposit", "fee", "acres",
                         "parking", "noi", "rsf", "spaces", "stories", "occupancy"]
        if any(hint in desc_lower for hint in numeric_hints) or any(hint in field_key for hint in numeric_hints):
            try:
                num = float(cleaned)
                if num == int(num) and "." not in cleaned:
                    return int(num)
                return num
            except ValueError:
                pass

        # Date coercion
        date_hints = ["date", "period", "start", "end", "expiration", "commencement", "execution"]
        if any(hint in desc_lower or hint in field_key for hint in date_hints):
            parsed_date = _try_parse_date(value)
            if parsed_date:
                return parsed_date

    # Numeric values that are strings
    if isinstance(value, str) and value.strip() in ("N/A", "n/a", "NA", "-", "--", "None", ""):
        return None

    return value


def _try_parse_date(val: str) -> Optional[str]:
    """Try to parse various date formats into YYYY-MM-DD."""
    import re
    from datetime import datetime

    val = val.strip()
    formats = [
        "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y",
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
        "%Y/%m/%d", "%m-%d-%Y", "%Y%m%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(val, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Try YYYY-MM format
    if re.match(r"^\d{4}-\d{2}$", val):
        return val  # Keep as-is for period fields

    return None


def _validate_and_clean_entries(
    entries: list[dict],
    schema: ExtractionSchema,
) -> list[dict[str, FieldResult]]:
    """Validate extracted entries, coerce types, apply null discipline."""
    cleaned = []

    for entry in entries:
        clean_entry = {}
        for field_key, field_def in schema.fields.items():
            if field_key in entry:
                raw = entry[field_key]
                if isinstance(raw, dict):
                    value = raw.get("value")
                    citation = raw.get("citation", "")
                    confidence = raw.get("confidence", 0.5)
                    page_numbers = raw.get("page_numbers")
                    source_locations = raw.get("source_locations")
                else:
                    value = raw
                    citation = ""
                    confidence = 0.5
                    page_numbers = None
                    source_locations = None
            else:
                value = None
                citation = ""
                confidence = 0.0
                page_numbers = None
                source_locations = None

            # Coerce value types
            if isinstance(value, list):
                value = [_coerce_value(v, field_key, field_def.description) for v in value]
            else:
                value = _coerce_value(value, field_key, field_def.description)

            # Null discipline: low confidence → null
            if value is not None and confidence < CONFIDENCE_NULL_THRESHOLD:
                logger.info(f"Nulling {field_key} due to low confidence ({confidence:.2f})")
                value = None
                citation = ""

            # Ensure citation is empty string for null values
            if value is None:
                citation = ""

            # Clamp confidence
            confidence = max(0.0, min(1.0, float(confidence)))

            clean_entry[field_key] = FieldResult(
                value=value,
                citation=str(citation),
                confidence=confidence,
                page_numbers=page_numbers,
                source_locations=source_locations,
            )

        cleaned.append(clean_entry)

    return cleaned


async def extract_document(
    file_path: str,
    filename: str,
    schema: ExtractionSchema,
    usage_stats: UsageStats,
    update_stage: Any = None,
) -> list[dict[str, FieldResult]]:
    """
    Main extraction entrypoint.
    Parse the document, then call LLM(s) to extract fields.
    Returns a list of entries (one per row or one for single-entry docs).
    """
    # ── Step 1: Parse ────────────────────────────────────────────────────────
    if update_stage:
        await update_stage("PARSING")

    doc_content = parse_document(file_path, filename)
    doc_type = doc_content["type"]
    doc_text = doc_content["text"]

    logger.info(
        f"Parsed {filename}: type={doc_type}, text_length={len(doc_text)}, "
        f"pages={len(doc_content.get('pages', []))}"
    )

    # ── Step 2: Choose model ─────────────────────────────────────────────────
    model = _pick_model(schema, doc_content)
    is_table = _is_table_schema(schema)

    logger.info(f"Using model={model}, is_table={is_table}, schema={schema.name}")

    if update_stage:
        await update_stage("EXTRACTING")

    # ── Step 3: Extract ──────────────────────────────────────────────────────
    if len(doc_text) <= MAX_SINGLE_CALL_CHARS:
        # Single call extraction
        entries = await _extract_single_call(schema, doc_content, model, is_table, usage_stats)
    else:
        # Chunked extraction
        entries = await _extract_chunked(schema, doc_content, model, is_table, usage_stats)

    # ── Step 4: Validate and clean ───────────────────────────────────────────
    if update_stage:
        await update_stage("VERIFYING")

    entries = _validate_and_clean_entries(entries, schema)

    # Ensure at least one entry
    if not entries:
        logger.warning(f"No entries extracted for {filename}, creating empty entry")
        empty_entry = {}
        for field_key in schema.fields:
            empty_entry[field_key] = FieldResult(value=None, citation="", confidence=0.0)
        entries = [empty_entry]

    logger.info(f"Extraction complete: {len(entries)} entries for {filename}")
    return entries


async def _extract_single_call(
    schema: ExtractionSchema,
    doc_content: dict,
    model: str,
    is_table: bool,
    usage_stats: UsageStats,
) -> list[dict]:
    """Extract all fields in a single LLM call."""
    doc_text = doc_content["text"]
    doc_type = doc_content["type"]

    if is_table:
        system_prompt, user_prompt = build_table_extraction_prompt(schema, doc_text, doc_type)
    else:
        system_prompt, user_prompt = build_extraction_prompt(schema, doc_text, doc_type, is_table)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response_text = await chat_completion(
        model=model,
        messages=messages,
        usage_stats=usage_stats,
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
    )

    parsed = _parse_json_response(response_text)
    entries = parsed.get("entries", [])

    if not entries:
        # Maybe the response is a single entry at the top level
        if any(k in parsed for k in schema.fields):
            entries = [parsed]
        # Or maybe it's wrapped differently
        elif "fields" in parsed:
            entries = [parsed["fields"]]
        elif "data" in parsed:
            entries = parsed["data"] if isinstance(parsed["data"], list) else [parsed["data"]]

    return entries


async def _extract_chunked(
    schema: ExtractionSchema,
    doc_content: dict,
    model: str,
    is_table: bool,
    usage_stats: UsageStats,
) -> list[dict]:
    """Extract from a document too large for a single call by chunking."""
    pages = doc_content.get("pages", [])
    doc_type = doc_content["type"]

    if is_table:
        # For tables, try to send the whole thing but with a cheaper model first
        # If it's too big, split by pages/sheets
        return await _extract_table_chunked(schema, doc_content, model, usage_stats)

    # Single-entry docs: extract from chunks and merge
    all_fields = {}
    chunk_size = 5  # pages per chunk

    for i in range(0, len(pages), chunk_size):
        chunk_pages = pages[i:i + chunk_size]
        chunk_text = "\n\n".join(
            f"[Page {p['page']}]\n{p['text']}" for p in chunk_pages
        )
        chunk_info = f"Pages {chunk_pages[0]['page']}-{chunk_pages[-1]['page']}"

        system_prompt, user_prompt = build_chunked_extraction_prompt(
            schema, chunk_text, chunk_info, doc_type,
            existing_values={k: v.get("value") for k, v in all_fields.items()} if all_fields else None,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_text = await chat_completion(
            model=model,
            messages=messages,
            usage_stats=usage_stats,
            temperature=0.0,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
        )

        parsed = _parse_json_response(response_text)
        fields = parsed.get("fields", parsed)

        # Merge: only overwrite if new value is non-null and higher confidence
        for k, v in fields.items():
            if k not in schema.fields:
                continue
            if isinstance(v, dict):
                new_val = v.get("value")
                new_conf = v.get("confidence", 0.5)
                if k not in all_fields or (new_val is not None and new_conf > all_fields[k].get("confidence", 0)):
                    all_fields[k] = v
            elif v is not None:
                if k not in all_fields or all_fields[k].get("value") is None:
                    all_fields[k] = {"value": v, "citation": "", "confidence": 0.5}

    # Fill missing fields with null
    for k in schema.fields:
        if k not in all_fields:
            all_fields[k] = {"value": None, "citation": "", "confidence": 0.0}

    return [all_fields]


async def _extract_table_chunked(
    schema: ExtractionSchema,
    doc_content: dict,
    model: str,
    usage_stats: UsageStats,
) -> list[dict]:
    """Handle large table documents by chunking pages/sheets."""
    pages = doc_content.get("pages", [])
    doc_type = doc_content["type"]
    all_entries = []

    # For XLSX, process sheets separately if needed
    if doc_type == "xlsx" and doc_content.get("sheets"):
        sheets = doc_content["sheets"]
        # Usually the main data is in the first sheet
        # Send each sheet as a chunk
        for sheet in sheets:
            chunk_text = sheet["text"]
            if len(chunk_text.strip()) < 100:
                continue  # Skip empty sheets

            system_prompt, user_prompt = build_table_extraction_prompt(
                schema, chunk_text, doc_type
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response_text = await chat_completion(
                model=model,
                messages=messages,
                usage_stats=usage_stats,
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )

            parsed = _parse_json_response(response_text)
            entries = parsed.get("entries", [])
            all_entries.extend(entries)

        if all_entries:
            return all_entries

    # For PDFs or fallback: chunk by pages
    chunk_size = 3
    for i in range(0, len(pages), chunk_size):
        chunk_pages = pages[i:i + chunk_size]
        chunk_text = "\n\n".join(
            f"[Page {p['page']}]\n{p['text']}" for p in chunk_pages
        )

        system_prompt, user_prompt = build_table_extraction_prompt(
            schema, chunk_text, doc_type
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_text = await chat_completion(
            model=model,
            messages=messages,
            usage_stats=usage_stats,
            temperature=0.0,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
        )

        parsed = _parse_json_response(response_text)
        entries = parsed.get("entries", [])
        all_entries.extend(entries)

    return all_entries

