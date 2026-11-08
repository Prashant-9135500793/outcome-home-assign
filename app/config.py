"""Configuration loaded from environment variables."""
import os

from dotenv import load_dotenv

# Load .env if present (local development; docker compose passes env vars explicitly)
load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


# ── LLM gateway (contract: all calls go through these two variables) ─────────
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

# ── Allowed models (assignment section 3) ────────────────────────────────────
MODEL_FLASH = "google/gemini-3.8-flash"            # fast, cheap, multimodal
MODEL_PRO = "google/gemini-3.1-pro-preview"        # stronger reasoning
MODEL_HAIKU = "anthropic/claude-haiku-5.5"         # fast, cheap
MODEL_SONNET = "anthropic/claude-sonnet-5.5"       # stronger reasoning
MODEL_GPT_FAST = "openai/gpt-6-luna"               # fast, cheap
MODEL_GPT_PRO = "openai/gpt-6.1-sol"               # stronger reasoning
MODEL_GLM_VISION = "z-ai/glm-5.3-flash"            # cheap vision model
MODEL_QWEN_VISION = "qwen/qwen3-vl-235b-a22b-instruct"  # open vision model
MODEL_DEEPSEEK = "deepseek/deepseek-v3.2"          # open text, strong reasoning

ALLOWED_MODELS = {
    MODEL_FLASH, MODEL_PRO, MODEL_HAIKU, MODEL_SONNET, MODEL_GPT_FAST,
    MODEL_GPT_PRO, MODEL_GLM_VISION, MODEL_QWEN_VISION, MODEL_DEEPSEEK,
}

# ── Model routing (overridable per env so we can A/B models in the eval loop) ─
# OpenRouter prices ($/1M in/out) at the time of writing: haiku/luna 0.10/0.50, glm-flash 0.03/0.08,
# gemini-flash 0.75/3.75, sonnet 2/10, gemini-pro 2/12. See DESIGN.md for the routing rationale.
# Model choice was measured on the sample set (DESIGN.md has the full A/B table).
# Table documents: many rows, mostly copying cells -> cheap model is enough (haiku: 1.00 on all 4 tables)
TABLE_MODEL = os.getenv("TABLE_MODEL", MODEL_HAIKU)
# Single-entry documents (leases, OMs): reading long prose, computing fields. Sonnet was the only
# model with zero errors on structured lease/OM fields; Gemini Pro, GPT-6.1-sol, Luna, Haiku had 3-7.
DOCUMENT_MODEL = os.getenv("DOCUMENT_MODEL", MODEL_SONNET)
# Transcribing scanned pages to text (vision)
OCR_MODEL = os.getenv("OCR_MODEL", MODEL_GLM_VISION)

# Reasoning ("thinking") tokens are billed as output. Copying cells and transcribing pages don't
# need them. Values: off | low | medium | high | default (leave it to the provider).
TABLE_REASONING = os.getenv("TABLE_REASONING", "low")  # low: fixes cell counting (t12 0.96 -> 1.00) for ~+$0.006/doc
DOCUMENT_REASONING = os.getenv("DOCUMENT_REASONING", "default")
OCR_REASONING = os.getenv("OCR_REASONING", "low")  # GLM-5.3-flash cannot disable reasoning

for _m in (TABLE_MODEL, DOCUMENT_MODEL, OCR_MODEL):
    if _m not in ALLOWED_MODELS:
        raise ValueError(f"Model {_m!r} is not on the allowed list")

# ── Service ──────────────────────────────────────────────────────────────────
PORT: int = _int("PORT", 8080)
WORKERS: int = _int("WORKERS", 5)
DB_PATH: str = os.getenv("DB_PATH", "jobs.db")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB

# ── LLM calls: concurrency, retry, rate limits ───────────────────────────────
LLM_CONCURRENCY_PER_MODEL: int = _int("LLM_CONCURRENCY_PER_MODEL", 16)
LLM_REQUEST_TIMEOUT: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "180"))
MAX_RETRIES: int = _int("LLM_MAX_RETRIES", 6)
RETRY_BASE_DELAY: float = 2.0
RETRY_MAX_DELAY: float = 60.0

# ── Extraction ───────────────────────────────────────────────────────────────
JOB_TIMEOUT_SECONDS: int = _int("JOB_TIMEOUT_SECONDS", 25 * 60)  # harness allows 30 min
CONFIDENCE_NULL_THRESHOLD: float = 0.3  # model-reported confidence below this -> null

# Single-entry documents up to this many characters go in one call (~100K tokens).
SINGLE_CALL_MAX_CHARS: int = _int("SINGLE_CALL_MAX_CHARS", 400_000)
# Table chunking: bounded by input size and by row records (which bounds output tokens).
TABLE_CHUNK_MAX_CHARS: int = _int("TABLE_CHUNK_MAX_CHARS", 30_000)
TABLE_CHUNK_MAX_RECORDS: int = _int("TABLE_CHUNK_MAX_RECORDS", 40)
# Text (PDF/DOCX) tables: rows carry long text, so chunks are smaller.
TEXT_TABLE_CHUNK_MAX_CHARS: int = _int("TEXT_TABLE_CHUNK_MAX_CHARS", 7_000)
MAX_OUTPUT_TOKENS: int = _int("MAX_OUTPUT_TOKENS", 16_000)
# OCR: concurrent page transcriptions per document
OCR_CONCURRENCY: int = _int("OCR_CONCURRENCY", 6)
