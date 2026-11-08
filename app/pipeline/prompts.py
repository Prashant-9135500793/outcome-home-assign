"""Prompt templates for LLM-based extraction.

Two output shapes:
  - single-entry documents (leases, OMs): {"fields": {key: {"value", "citation", "confidence"}}}
  - table chunks: {"document": {key: {"value", "citation"}}, "rows": [{"_row"/"_cite", key: value}]}
    Row values are bare (no per-cell citation objects) to keep output tokens ~3x lower; the
    extractor attaches citations afterwards (cell lookup for spreadsheets, row cite for text).
"""
from __future__ import annotations

import re

from app.models import ExtractionSchema

# Field descriptions that mark a value as document-wide in a table schema
_DOC_LEVEL_RE = re.compile(
    r"same (value )?for (every|each|all) (row|entr|line|unit)|same in every (row|entry)|applies to the whole document",
    re.IGNORECASE,
)


def doc_level_keys(schema: ExtractionSchema) -> list[str]:
    """Fields whose value is the same for every row (header values like property name, period)."""
    return [k for k, f in schema.fields.items() if not f.primary and _DOC_LEVEL_RE.search(f.description or "")]


def primary_key(schema: ExtractionSchema) -> str | None:
    return next((k for k, f in schema.fields.items() if f.primary), None)


def fields_block(schema: ExtractionSchema, keys: list[str] | None = None) -> str:
    keys = keys if keys is not None else list(schema.fields)
    lines = []
    for k in keys:
        f = schema.fields[k]
        line = f"- `{k}` ({f.name}): {f.description}"
        if f.primary:
            line += "  [ROW IDENTIFIER]"
        lines.append(line)
    return "\n".join(lines)


VALUE_RULES = """VALUE FORMAT
- Numbers: JSON numbers only (1425.5), never strings, no currency symbols, no thousands separators. Negative numbers stay negative; "(1,234)" means -1234.
- Percentages: the number exactly as the document prints it: "93.75%" -> 93.75, "0.91%" -> 0.91.
- Dates: "YYYY-MM-DD". A month/year-only field uses the format its description asks for.
- Booleans: true / false.
- Lists: JSON arrays, in document order. A list field with nothing to list is [].
- Text: copy it as written (trim surrounding whitespace). Do not paraphrase fields that say "exactly as shown" or "verbatim".
- Short-phrase fields ("in a short phrase"): answer concisely using the document's own key words, no parenthetical explanations.
- Identifiers and names: copy the complete value that follows its label, with every part, space and punctuation mark (e.g. "Property =  417 .417" -> "417 .417", not "417"). Drop only the label itself.

NULL DISCIPLINE (very important)
- If the document does not contain the value, return null. A wrong value is penalised far more than a null.
- Never fill a value from general knowledge, typical market terms, or another field's value.
- Calculated fields ARE expected: when a description says to compute a value (a sum, a count, a difference, a duration), compute it carefully from the document's own figures. Double-check arithmetic.
- If a description says to use 0, false or [] when something is absent, follow it (that is not a guess).
- Blank cells: when the cell/column for a field is empty for a record, the value is null - not 0 - unless the description says to use 0 when absent. Never take a value from a neighbouring column."""


# ── Single-entry documents ───────────────────────────────────────────────────
def build_single_entry_prompt(schema: ExtractionSchema, document_text: str, doc_type: str,
                              part_note: str = "") -> tuple[str, str]:
    system = f"""You extract structured data from commercial real estate documents with high precision.

DOCUMENT TYPE: {schema.name}{(' - ' + schema.description) if schema.description else ''}

The document describes ONE subject, so return ONE set of fields.

{VALUE_RULES}

CITATIONS
- "citation": a short verbatim snippet (under 200 characters) from the document that supports the value, prefixed with its location, e.g. "[Page 3] Base Rent: $4,250.00 per month". For a computed value, cite the main figure(s) you used.
- Use "" as the citation when the value is null.
- "confidence": your probability (0.0-1.0) that the value is exactly right. Use < 0.5 when the document is ambiguous.

Read the whole document before answering: amendments, exhibits, riders and handwritten changes can override earlier text. Use the governing (latest/amended) figure.

FIELDS
{fields_block(schema)}

OUTPUT: one JSON object, no prose:
{{"fields": {{"<field_key>": {{"value": ..., "citation": "...", "confidence": 0.0}}, ...}}}}
Include every field key listed above exactly once."""

    user = f"""{part_note}FILE FORMAT: {doc_type.upper()}

<document>
{document_text}
</document>

Return the JSON object now."""
    return system, user


# ── Table chunks ─────────────────────────────────────────────────────────────
def build_table_chunk_prompt(
    schema: ExtractionSchema,
    doc_keys: list[str],
    row_keys: list[str],
    chunk_text: str,
    context_text: str,
    chunk_label: str,
    is_grid: bool,
) -> tuple[str, str]:
    pk = primary_key(schema)
    if is_grid:
        loc_rule = ('- "_row": the spreadsheet row number (the r-number in the first column) of the line that '
                    'starts this row/record, as an integer.')
        layout = ("The spreadsheet is shown as a table whose first column is the real row number (r8 = row 8) "
                  "and whose header gives the real column letters. A record may span several consecutive lines "
                  "(e.g. a unit line followed by its charge lines); fold each record into one row.")
        row_example = '{"_row": 8, ' + ", ".join(f'"{k}": ...' for k in row_keys[:3]) + ", ...}"
    else:
        loc_rule = ('- "_cite": a short verbatim snippet identifying the row, prefixed with its location, '
                    'e.g. "[Page 2] 6101-0000 Base Rent 1,250,000".')
        layout = "The document text is split into pages marked [Page N]; tables may be rendered as text or markdown."
        row_example = '{"_cite": "[Page 1] ...", ' + ", ".join(f'"{k}": ...' for k in row_keys[:3]) + ", ...}"

    doc_section = ""
    if doc_keys:
        doc_section = f"""
DOCUMENT-LEVEL FIELDS (one value for the whole document; usually in the report header, which may be in the CONTEXT section):
{fields_block(schema, doc_keys)}
"""

    system = f"""You extract table rows from commercial real estate documents with high precision.

DOCUMENT TYPE: {schema.name}{(' - ' + schema.description) if schema.description else ''}

{layout}
You are given ONE PART of the document ({chunk_label}). Extract every qualifying row that appears in the PART section, and nothing from the CONTEXT section (context is only there so you can read headers, column titles and section names).
{doc_section}
ROW-LEVEL FIELDS (one row per record; `{pk}` identifies the row):
{fields_block(schema, row_keys)}

ROW RULES
- Which lines count as rows is defined ONLY by the document type description and the field descriptions above. Apply that definition literally to every line, including total and subtotal lines: if a total line meets the definition (for example it has its own identifier and an amount), it IS a row; if the definition excludes it, skip it. Do not apply your own idea of what a "detail line" is.
- Do not skip rows and do not invent rows. Keep the document's order.
- Every row must contain every row-level field key; use null where the record has no value.
- Apply each field description literally. "The nearest header above the line" means the closest header line physically above it in the document, even for a total line whose logical parent is a higher-level heading.
- Counts over cells (e.g. "number of months with a non-zero value"): go through the cells one by one; a cell showing 0, 0.00 or blank does not count.
- Classification fields: a line that is a computed result mixing revenue and expenses (net operating income, net income, cash flow) is neither revenue nor expense.
{loc_rule}

{VALUE_RULES}

OUTPUT: one JSON object, no prose:
{{"document": {{{", ".join(f'"{k}": {{"value": ..., "citation": "..."}}' for k in doc_keys) if doc_keys else ''}}},
 "rows": [{row_example}]}}
If the PART contains no qualifying rows, return "rows": []."""

    user = ""
    if context_text.strip():
        user += f"<context>\n{context_text}\n</context>\n\n"
    user += f"<part label=\"{chunk_label}\">\n{chunk_text}\n</part>\n\nReturn the JSON object now."
    return system, user
