"""Prompt templates for LLM-based extraction."""
from __future__ import annotations

from app.models import ExtractionSchema


def build_extraction_prompt(
    schema: ExtractionSchema,
    document_text: str,
    doc_type: str,
    is_table: bool,
) -> str:
    """Build the system + user prompt for extraction."""

    # Build field descriptions
    field_descriptions = []
    primary_field = None
    for key, f in schema.fields.items():
        desc = f"- **{key}** ({f.name}): {f.description}"
        if f.primary:
            desc += " [PRIMARY KEY - one per row]"
            primary_field = key
        field_descriptions.append(desc)

    fields_block = "\n".join(field_descriptions)

    if is_table:
        entry_instruction = (
            "This document contains a TABLE with multiple rows. "
            "Return ONE entry (JSON object) per row in the table. "
            "Document-level fields (like property name, report date) must be REPEATED "
            "with the same value in EVERY entry. "
            f"The primary key field is `{primary_field}` — use it to identify each unique row."
        )
    else:
        entry_instruction = (
            "This document has NO table. Return exactly ONE entry with all fields."
        )

    system_prompt = f"""You are a precise document data extraction system for commercial real estate documents.

TASK: Extract ALL specified fields from the document below. Return a JSON object.

RULES — FOLLOW EXACTLY:
1. Return valid JSON with a single key "entries" containing a list of objects.
2. Each entry must contain EVERY field listed below, no exceptions.
3. Each field value must be an object with exactly these keys:
   - "value": the extracted value (number, string, date, list, boolean, or null)
   - "citation": a short verbatim quote from the document showing where you found it, prefixed with location like "[Page 3] Base Rent: $4,250.00" or "[Sheet1!A5] value". Use empty string "" when value is null.
   - "confidence": a float 0.0-1.0 estimating correctness.

4. FORMATTING:
   - Numbers: JSON numbers only. NO dollar signs, NO commas, NO quotes around numbers. Example: 1425, not "$1,425" or "1425"
   - Dates: "YYYY-MM-DD" format always. Example: "2024-01-15"
   - Percentages: as the document prints them. If it says "93.75%", the value is 93.75 (not 0.9375)
   - Booleans: true or false (JSON booleans)
   - Lists: JSON arrays. For rent schedule lists, maintain the same order as the document.
   - Null: use JSON null when the value is genuinely NOT in the document.

5. NULL DISCIPLINE — CRITICAL:
   - If a field is not mentioned in the document, return null. DO NOT GUESS.
   - A wrong value is MUCH worse than null. When in doubt, return null.
   - Set confidence < 0.5 for uncertain values and null for values you cannot find.

6. CITATIONS — IMPORTANT:
   - Always include the location: [Page N] for PDFs, [Sheet: Name, Cell: XX] or [SheetName!CellRef] for spreadsheets
   - Quote the actual text/numbers from the source
   - Empty string "" for null values

{entry_instruction}

FIELDS TO EXTRACT:
{fields_block}
"""

    user_prompt = f"""DOCUMENT TYPE: {schema.name} — {schema.description}
FILE FORMAT: {doc_type.upper()}

DOCUMENT CONTENT:
{document_text}

Extract ALL fields now. Return ONLY valid JSON, no explanation. Format:
{{"entries": [{{{", ".join(f'"{k}": {{"value": ..., "citation": "...", "confidence": ...}}' for k in list(schema.fields.keys())[:3])}...}}]}}"""

    return system_prompt, user_prompt


def build_table_extraction_prompt(
    schema: ExtractionSchema,
    document_text: str,
    doc_type: str,
) -> str:
    """Specialized prompt for large table documents (rent rolls, T12s)."""

    # Identify primary key and doc-level vs row-level fields
    primary_field = None
    doc_fields = []
    row_fields = []
    for key, f in schema.fields.items():
        if f.primary:
            primary_field = key
            row_fields.append((key, f))
        elif any(kw in f.description.lower() for kw in ["same for every", "same value", "property name", "report", "period"]):
            doc_fields.append((key, f))
        else:
            row_fields.append((key, f))

    doc_fields_desc = "\n".join(
        f"- **{k}** ({f.name}): {f.description}" for k, f in doc_fields
    )
    row_fields_desc = "\n".join(
        f"- **{k}** ({f.name}): {f.description}" for k, f in row_fields
    )

    system_prompt = f"""You are a precise data extraction system for commercial real estate documents containing tables.

TASK: Extract data from a tabular document. The document has:
1. DOCUMENT-LEVEL fields (same value in every row):
{doc_fields_desc}

2. ROW-LEVEL fields (one value per row, primary key: {primary_field}):
{row_fields_desc}

RULES:
- Return valid JSON: {{"entries": [...]}}
- ONE entry per row in the table.
- Every entry must have ALL fields (both document-level and row-level).
- Document-level fields repeat the same value in every entry.
- Numbers as JSON numbers (no $, no commas). Dates as "YYYY-MM-DD". Percentages as printed.
- null when a value is not in the document. DO NOT GUESS.
- Each field: {{"value": ..., "citation": "...", "confidence": 0.0-1.0}}
- Citations: include location ([Page N] or [Sheet!Cell]) and verbatim text. "" for nulls.
"""

    user_prompt = f"""DOCUMENT TYPE: {schema.name} — {schema.description}
FILE: {doc_type.upper()}

DOCUMENT:
{document_text}

Extract ALL rows and ALL fields. Return ONLY valid JSON."""

    return system_prompt, user_prompt


def build_chunked_extraction_prompt(
    schema: ExtractionSchema,
    chunk_text: str,
    chunk_info: str,
    doc_type: str,
    existing_values: dict | None = None,
) -> str:
    """Prompt for extracting from a chunk of a large document."""

    field_descriptions = []
    for key, f in schema.fields.items():
        desc = f"- **{key}** ({f.name}): {f.description}"
        if f.primary:
            desc += " [PRIMARY KEY]"
        field_descriptions.append(desc)

    fields_block = "\n".join(field_descriptions)

    context = ""
    if existing_values:
        context = f"\nVALUES ALREADY EXTRACTED (from other chunks, do not override unless you find a better match):\n{existing_values}\n"

    system_prompt = f"""You are a precise extraction system. You are processing ONE CHUNK of a larger document.
Extract any values you can find for the fields below. Return null for fields not found in THIS chunk.

RULES:
- Valid JSON: {{"fields": {{"field_key": {{"value": ..., "citation": "...", "confidence": 0.0-1.0}}, ...}}}}
- Numbers as JSON numbers, dates as YYYY-MM-DD, percentages as printed.
- null when not found in this chunk. DO NOT GUESS.
- Citations: include [Page N] or [Sheet!Cell] prefix.
{context}

FIELDS:
{fields_block}
"""

    user_prompt = f"""CHUNK: {chunk_info}
DOCUMENT TYPE: {schema.name}

CONTENT:
{chunk_text}

Extract what you can. Return ONLY valid JSON."""

    return system_prompt, user_prompt

