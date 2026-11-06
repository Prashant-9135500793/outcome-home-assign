"""OpenAI-compatible LLM client with retry, backoff, and usage tracking."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI, APIStatusError, APITimeoutError, APIConnectionError

from app.config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
)

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Accumulates token/cost stats across multiple calls."""
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    per_model: dict[str, dict] = field(default_factory=dict)

    def record(self, model: str, tokens_in: int, tokens_out: int, cost: float = 0.0):
        self.model_calls += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += cost
        if model not in self.per_model:
            self.per_model[model] = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
        self.per_model[model]["calls"] += 1
        self.per_model[model]["tokens_in"] += tokens_in
        self.per_model[model]["tokens_out"] += tokens_out
        self.per_model[model]["cost_usd"] += cost

    def to_metadata(self) -> dict:
        return {
            "model_calls": self.model_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
            "models": self.per_model,
        }


# ── Pricing table (approximate $/1M tokens) ─────────────────────────────────
_PRICING = {
    "google/gemini-3.8-flash":            {"input": 0.10,  "output": 0.40},
    "google/gemini-3.1-pro-preview":      {"input": 1.25,  "output": 5.00},
    "anthropic/claude-haiku-5.5":         {"input": 0.80,  "output": 4.00},
    "anthropic/claude-sonnet-5.5":        {"input": 3.00,  "output": 15.00},
    "openai/gpt-6-luna":                  {"input": 0.50,  "output": 2.00},
    "openai/gpt-6.1-sol":                 {"input": 2.50,  "output": 10.00},
    "z-ai/glm-5.3-flash":                {"input": 0.05,  "output": 0.20},
    "qwen/qwen3-vl-235b-a22b-instruct":  {"input": 0.30,  "output": 1.20},
    "deepseek/deepseek-v3.2":            {"input": 0.25,  "output": 1.00},
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = _PRICING.get(model, {"input": 1.0, "output": 4.0})
    return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000


# ── Rate limiter (simple semaphore per model) ────────────────────────────────
_model_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_semaphore(model: str) -> asyncio.Semaphore:
    if model not in _model_semaphores:
        # Allow 3 concurrent calls per model to avoid 429s
        _model_semaphores[model] = asyncio.Semaphore(3)
    return _model_semaphores[model]


# ── Singleton client ─────────────────────────────────────────────────────────
_client: Optional[AsyncOpenAI] = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=120.0,
            max_retries=0,  # we handle retries ourselves
        )
    return _client


async def chat_completion(
    model: str,
    messages: list[dict],
    usage_stats: UsageStats,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    response_format: Optional[dict] = None,
) -> str:
    """Call the LLM with retry and backoff. Returns the text content."""
    client = get_client()
    sem = _get_semaphore(model)

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with sem:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                resp = await client.chat.completions.create(**kwargs)

            # Extract usage
            tok_in = getattr(resp.usage, "prompt_tokens", 0) or 0
            tok_out = getattr(resp.usage, "completion_tokens", 0) or 0
            cost = _estimate_cost(model, tok_in, tok_out)
            usage_stats.record(model, tok_in, tok_out, cost)

            content = resp.choices[0].message.content or ""
            logger.info(
                f"LLM call OK: model={model} tokens_in={tok_in} tokens_out={tok_out} "
                f"cost=${cost:.4f} attempt={attempt+1}"
            )
            return content

        except APIStatusError as e:
            if e.status_code == 429:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                # Check for Retry-After header
                retry_after = None
                if hasattr(e, 'response') and e.response is not None:
                    retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(float(retry_after), delay)
                    except ValueError:
                        pass
                logger.warning(
                    f"Rate limited (429) on {model}, attempt {attempt+1}/{MAX_RETRIES+1}, "
                    f"retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue
            elif e.status_code >= 500:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                logger.warning(
                    f"Server error ({e.status_code}) on {model}, attempt {attempt+1}, "
                    f"retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue
            else:
                logger.error(f"LLM API error: {e.status_code} {e.message}")
                raise

        except (APITimeoutError, APIConnectionError) as e:
            delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
            logger.warning(
                f"Connection error on {model}, attempt {attempt+1}, retrying in {delay:.1f}s: {e}"
            )
            await asyncio.sleep(delay)
            continue

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES+1} attempts for model {model}")

