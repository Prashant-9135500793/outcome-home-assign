"""OCR for scanned PDF pages: render each page to an image and have a vision model transcribe it.

We transcribe to text first (rather than extracting fields straight from images) because:
  - each page is an independent, cheap call that can run in parallel;
  - the extraction step then sees the whole document at once, with [Page N] markers for citations;
  - the same extraction code path serves digital and scanned documents.
"""
from __future__ import annotations

import asyncio
import logging

from app import config
from app.llm.client import UsageStats, chat_completion
from app.pipeline.parser import join_pages, render_pdf_page_jpeg_base64

logger = logging.getLogger(__name__)

OCR_PROMPT = """Transcribe this scanned document page to plain text, exactly as written.

Rules:
- Reproduce every word and number verbatim, in reading order. Do not summarise, correct or omit anything.
- Keep headings, article/section numbers and paragraph breaks.
- Render tables as markdown tables, one table row per line.
- Handwritten text: transcribe it and mark it as [handwritten: ...].
- Crossed-out text: mark it as [struck: ...]. If a struck value is replaced by a handwritten one, keep both markers next to each other.
- Blank form fields: write [blank]. Signatures: write [signature]. Illegible text: write [illegible].
- Output only the transcription, with no commentary."""


async def _ocr_page(file_path: str, page_number: int, usage: UsageStats, sem: asyncio.Semaphore) -> str:
    async with sem:
        image_b64 = await asyncio.to_thread(render_pdf_page_jpeg_base64, file_path, page_number)
        if not image_b64:
            return ""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }]
        try:
            resp = await chat_completion(
                model=config.OCR_MODEL, messages=messages, usage_stats=usage,
                temperature=0.0, max_tokens=6000, reasoning=config.OCR_REASONING,
            )
        except Exception as e:  # noqa: BLE001 - one bad page should not fail the document
            logger.error(f"OCR failed for page {page_number}: {e}")
            return ""
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        return text


async def ocr_scanned_pages(doc_content: dict, file_path: str, usage: UsageStats) -> dict:
    """Replace the text of scanned pages with a vision-model transcription. Mutates and returns doc_content."""
    scanned = doc_content.get("scanned_pages") or []
    if not scanned:
        return doc_content

    logger.info(f"OCR: transcribing {len(scanned)} scanned page(s) with {config.OCR_MODEL}")
    sem = asyncio.Semaphore(config.OCR_CONCURRENCY)
    texts = await asyncio.gather(*(_ocr_page(file_path, p, usage, sem) for p in scanned))

    by_page = dict(zip(scanned, texts))
    failed = [p for p, t in by_page.items() if not t]
    for page in doc_content["pages"]:
        t = by_page.get(page["page"])
        if t:
            # Keep whatever the text layer had (e.g. a stamped header) after the transcription
            page["text"] = t + ("\n" + page["text"] if page["text"].strip() else "")
            page["ocr"] = True
    doc_content["text"] = join_pages(doc_content["pages"])
    doc_content["ocr_failed_pages"] = failed
    if failed:
        logger.warning(f"OCR produced no text for pages {failed}")
    return doc_content
