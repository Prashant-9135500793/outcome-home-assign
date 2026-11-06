"""Configuration loaded from environment variables."""
import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

# ── Models ───────────────────────────────────────────────────────────────────
MODEL_FLASH = "google/gemini-3.8-flash"           # fast, cheap, multimodal
MODEL_PRO = "google/gemini-3.1-pro-preview"        # stronger reasoning
MODEL_HAIKU = "anthropic/claude-haiku-5.5"         # fast, cheap
MODEL_SONNET = "anthropic/claude-sonnet-5.5"       # stronger reasoning
MODEL_GPT_FAST = "openai/gpt-6-luna"               # fast, cheap
MODEL_GPT_PRO = "openai/gpt-6.1-sol"              # stronger reasoning
MODEL_GLM_VISION = "z-ai/glm-5.3-flash"           # cheap vision model
MODEL_QWEN_VISION = "qwen/qwen3-vl-235b-a22b-instruct"  # open vision model
MODEL_DEEPSEEK = "deepseek/deepseek-v3.2"          # open text, strong reasoning

# ── Default model routing ────────────────────────────────────────────────────
# Cheap model for structured / simple docs
DEFAULT_MODEL = MODEL_FLASH
# Stronger model for complex reasoning (leases, OMs)
REASONING_MODEL = MODEL_SONNET
# Vision model for scanned PDFs
VISION_MODEL = MODEL_GLM_VISION

# ── Service ──────────────────────────────────────────────────────────────────
PORT: int = int(os.getenv("PORT", "8080"))
WORKERS: int = int(os.getenv("WORKERS", "5"))
DB_PATH: str = os.getenv("DB_PATH", "jobs.db")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB

# ── Retry / rate-limit ───────────────────────────────────────────────────────
MAX_RETRIES: int = 5
RETRY_BASE_DELAY: float = 2.0
RETRY_MAX_DELAY: float = 60.0

# ── Extraction ───────────────────────────────────────────────────────────────
JOB_TIMEOUT_SECONDS: int = 25 * 60  # 25 min internal (30 min external)
CONFIDENCE_NULL_THRESHOLD: float = 0.3  # below this, return null instead

