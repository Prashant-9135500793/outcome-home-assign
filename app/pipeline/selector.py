"""Evidence page selection for single-entry documents (cost control).

Leases and OMs are read by DOCUMENT_MODEL (Sonnet, $2/M input), and its input is most of the
per-document cost. The answers sit on few pages: the sample answer key cites 8 of 25 pages
(lease_001), 11 of 21 (lease_003) and 6 of 22 (om_001). So a cheap model (SELECT_MODEL, Haiku at
$0.10/M, ~20x cheaper) first reads the whole document and names, per field, the pages that state or
change the value. DOCUMENT_MODEL then reads only those pages plus the opening pages.

Recall is protected on several sides, because a page the selector misses becomes a null:
  - the selector is told to over-include (amendments, riders, exhibits, schedules, definitions);
  - the opening pages (parties, dates, premises, reference page) are always kept;
  - each kept page also carries the edges of its neighbours, so a clause that runs over a page
    break is still whole;
  - if the selection would still be most of the document, or the call fails, the full document
    is used (the old behaviour). Short documents skip selection: there is little to save.

DOCX files have no pages, so they are cut into ~UNIT_CHARS blocks on paragraph boundaries; the
blocks only serve the selection and are not shown as page numbers to DOCUMENT_MODEL.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app import config
from app.llm.client import UsageStats, chat_completion
from app.models import ExtractionSchema
from app.pipeline.prompts import fields_block

logger = logging.getLogger(__name__)

UNIT_CHARS = 3500            # block size for documents without pages (DOCX)
ALWAYS_KEEP = 2              # opening units always kept
NEIGHBOUR_EDGE_CHARS = 300   # chars of each neighbour's facing edge kept with a selected unit


@dataclass
class Unit:
    label: str      # "Page 7" or "Block 3"
    text: str       # text as DOCUMENT_MODEL will see it (includes the [Page N] marker for pages)
    body: str       # text without marker


def build_units(doc: dict) -> list[Unit]:
    pages = doc.get("pages") or []
    if len(pages) > 1:
        return [Unit(f"Page {p['page']}", f"[Page {p['page']}]\n{p['text']}", p["text"]) for p in pages]
    # One "page" (DOCX): split on paragraph boundaries
    text = doc.get("text", "")
    units: list[Unit] = []
    cur: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        if cur and size + len(para) > UNIT_CHARS:
            body = "\n\n".join(cur)
            units.append(Unit(f"Block {len(units) + 1}", body, body))
            cur, size = [], 0
        cur.append(para)
        size += len(para) + 2
    if cur:
        body = "\n\n".join(cur)
        units.append(Unit(f"Block {len(units) + 1}", body, body))
    return units


def build_selection_prompt(schema: ExtractionSchema, units: list[Unit]) -> tuple[str, str]:
    kind = units[0].label.split()[0].lower() if units else "page"
    system = f"""You are the first stage of a document extraction pipeline. A second, more expensive reader will \
extract the fields below, but it will ONLY see the {kind}s you select. A {kind} you leave out is invisible to it.

DOCUMENT TYPE: {schema.name}{(' - ' + schema.description) if schema.description else ''}

FIELDS
{fields_block(schema)}

TASK: for every field, list the {kind} numbers where its value is actually stated (or, for a computed
field, where the figures it is computed from are stated). Usually that is one or two {kind}s per field. When the
same fact is stated on several {kind}s, prefer the definitive statement (a reference page, rent schedule or the
operative clause) over passing mentions. When the clause that states a value runs over onto the next
{kind}, list both {kind}s. A field whose value the document does not state gets [].

Separately, under "overrides", list every {kind} holding an amendment, rider, addendum, side letter, or a
handwritten or struck change that alters a value of one of the fields.

OUTPUT: one JSON object, no prose:
{{"pages": {{"<field_key>": [<{kind} numbers>], ...}}, "overrides": [<{kind} numbers>]}}"""
    body = "\n\n".join(f"<<{u.label}>>\n{u.body}" for u in units)
    user = f"<document>\n{body}\n</document>\n\nReturn the JSON object now."
    return system, user


def _assemble(units: list[Unit], keep: set[int]) -> str:
    """Selected units in order; neighbours of selected units contribute their facing edges."""
    out: list[str] = []
    for i, u in enumerate(units):
        if i in keep:
            if i - 1 >= 0 and i - 1 not in keep:
                prev = units[i - 1].body
                out.append(f"[...end of {units[i - 1].label}]\n{prev[-NEIGHBOUR_EDGE_CHARS:]}")
            out.append(u.text)
            if i + 1 < len(units) and i + 1 not in keep:
                nxt = units[i + 1].body
                out.append(f"[start of {units[i + 1].label}...]\n{nxt[:NEIGHBOUR_EDGE_CHARS]}")
    return "\n\n".join(out)


async def select_text(schema: ExtractionSchema, doc: dict, usage: UsageStats,
                      parse_json) -> tuple[str, dict]:
    """Return (text for DOCUMENT_MODEL, info). Falls back to the full text whenever unsure."""
    full = doc["text"]
    info: dict = {"selection": "full"}
    if not config.DOCUMENT_PAGE_SELECTION or len(full) < config.SELECTION_MIN_CHARS:
        return full, info
    units = build_units(doc)
    if len(units) <= ALWAYS_KEEP + 1:
        return full, info

    system, user = build_selection_prompt(schema, units)
    try:
        resp = await chat_completion(
            model=config.SELECT_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            usage_stats=usage, max_tokens=12000, response_format={"type": "json_object"},
            reasoning=config.SELECT_REASONING,
        )
        parsed = parse_json(resp.content)
        picked = parsed.get("pages")
        if not isinstance(picked, dict):
            raise ValueError("no 'pages' object")
        overrides = parsed.get("overrides")
        picked = {**picked, "_overrides": overrides if isinstance(overrides, list) else []}
    except Exception as e:  # noqa: BLE001 - selection is an optimisation; the full text always works
        logger.warning(f"Page selection failed ({e}); using the full document")
        return full, info

    by_number = {int(re.sub(r"\D", "", u.label)): i for i, u in enumerate(units)}
    keep = set(range(ALWAYS_KEEP))
    for vals in picked.values():
        for v in vals if isinstance(vals, list) else [vals]:
            try:
                idx = by_number.get(int(v))
            except (TypeError, ValueError):
                continue
            if idx is not None:
                keep.add(idx)

    number_of = {i: n for n, i in by_number.items()}
    logger.info(f"Page selection picks: {sorted(number_of[i] for i in keep)}")
    text = _assemble(units, keep)
    share = len(text) / max(1, len(full))
    info = {"selection": f"{len(keep)}/{len(units)} {units[0].label.split()[0].lower()}s",
            "share": round(share, 2)}
    if share > config.SELECTION_MAX_SHARE:
        logger.info(f"Page selection kept {share:.0%} of the text; using the full document")
        return full, {"selection": "full", "share": round(share, 2)}
    logger.info(f"Page selection: {info['selection']} kept ({share:.0%} of the text)")
    return text, info
