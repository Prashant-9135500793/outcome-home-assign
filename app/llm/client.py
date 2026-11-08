"""OpenAI-compatible LLM client with retry, backoff, concurrency limits and usage tracking."""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app import config

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Accumulates token/cost stats across all calls made for one job."""
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
        m = self.per_model.setdefault(model, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
        m["calls"] += 1
        m["tokens_in"] += tokens_in
        m["tokens_out"] += tokens_out
        m["cost_usd"] = round(m["cost_usd"] + cost, 6)

    def to_metadata(self) -> dict:
        return {
            "model_calls": self.model_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
            "models": self.per_model,
        }


@dataclass
class LLMResponse:
    content: str
    finish_reason: Optional[str]
    tokens_in: int
    tokens_out: int
    cost_usd: float

    @property
    def truncated(self) -> bool:
        return self.finish_reason in ("length", "max_tokens")


# Fallback price table ($ per 1M tokens, OpenRouter list prices), used only when the gateway
# does not report the real cost in the response.
_FALLBACK_PRICING = {
    "google/gemini-3.8-flash": (0.75, 3.75),
    "google/gemini-3.1-pro-preview": (2.00, 12.00),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "openai/gpt-6-luna": (0.10, 0.50),
    "openai/gpt-6.1-sol": (2.00, 10.00),
    "z-ai/glm-5.3-flash": (0.03, 0.08),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.21, 1.90),
    "deepseek/deepseek-v3.2": (0.26, 0.38),
}


def reasoning_param(mode: str) -> Optional[dict]:
    """Map a config value to OpenRouter's unified `reasoning` request parameter."""
    mode = (mode or "default").lower()
    if mode == "off":
        return {"enabled": False}
    if mode in ("low", "medium", "high"):
        return {"effort": mode}
    return None


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = _FALLBACK_PRICING.get(model, (1.0, 4.0))
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


def _reported_cost(usage: Any) -> Optional[float]:
    """OpenRouter returns the real charge as usage.cost when usage accounting is requested."""
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    if cost is None:
        extra = getattr(usage, "model_extra", None) or {}
        cost = extra.get("cost")
    try:
        return float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None


# ── Concurrency: one semaphore per (event loop, model) ───────────────────────
_semaphores: dict[tuple[int, str], asyncio.Semaphore] = {}


def _get_semaphore(model: str) -> asyncio.Semaphore:
    key = (id(asyncio.get_running_loop()), model)
    if key not in _semaphores:
        _semaphores[key] = asyncio.Semaphore(config.LLM_CONCURRENCY_PER_MODEL)
    return _semaphores[key]


_client: Optional[AsyncOpenAI] = None
# Models that rejected a reasoning setting with HTTP 400 (e.g. "reasoning is mandatory"): we stop
# sending it for the rest of the process instead of paying a failed round trip on every call.
_NO_REASONING_CONTROL: set[str] = set()


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            timeout=config.LLM_REQUEST_TIMEOUT,
            max_retries=0,  # retries are handled below so we control backoff and logging
        )
    return _client


def _backoff(attempt: int) -> float:
    delay = min(config.RETRY_BASE_DELAY * (2 ** attempt), config.RETRY_MAX_DELAY)
    return delay * (0.75 + random.random() * 0.5)  # jitter so parallel chunks don't retry in lockstep


class LLMError(RuntimeError):
    """Raised when a call fails permanently (non-retryable status, or retries exhausted)."""


async def chat_completion(
    model: str,
    messages: list[dict],
    usage_stats: UsageStats,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    response_format: Optional[dict] = None,
    reasoning: str = "default",
) -> LLMResponse:
    """Call the model with retry/backoff on 429, 5xx, timeouts and empty responses."""
    if model not in config.ALLOWED_MODELS:
        raise LLMError(f"Model {model!r} is not on the allowed list")

    client = get_client()
    sem = _get_semaphore(model)
    last_error: Optional[str] = None
    reasoning_cfg = None if model in _NO_REASONING_CONTROL else reasoning_param(reasoning)

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": {"usage": {"include": True}},
            }
            if response_format:
                kwargs["response_format"] = response_format
            if reasoning_cfg is not None:
                kwargs["extra_body"]["reasoning"] = reasoning_cfg
            async with sem:
                resp = await client.chat.completions.create(**kwargs)

            if not resp.choices:
                # OpenRouter sometimes returns 200 with an error body and no choices
                err = getattr(resp, "error", None) or (resp.model_extra or {}).get("error")
                raise _TransientError(f"no choices in response (error={err})")

            usage = resp.usage
            tok_in = getattr(usage, "prompt_tokens", 0) or 0
            tok_out = getattr(usage, "completion_tokens", 0) or 0
            cost = _reported_cost(usage)
            if cost is None:
                cost = _estimate_cost(model, tok_in, tok_out)
            usage_stats.record(model, tok_in, tok_out, cost)

            choice = resp.choices[0]
            content = choice.message.content or ""
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_toks = getattr(details, "reasoning_tokens", 0) or 0
            logger.info(
                f"LLM call OK: model={model} in={tok_in} out={tok_out} (reasoning={reasoning_toks}) cost=${cost:.5f} "
                f"finish={choice.finish_reason} attempt={attempt + 1}"
            )
            if not content.strip() and choice.finish_reason not in ("length", "max_tokens"):
                raise _TransientError("empty completion")
            return LLMResponse(content, choice.finish_reason, tok_in, tok_out, cost)

        except APIStatusError as e:
            status = e.status_code
            last_error = f"HTTP {status}: {e.message}"
            if status == 429 or status >= 500 or status == 408:
                delay = _backoff(attempt)
                retry_after = e.response.headers.get("Retry-After") if e.response is not None else None
                if retry_after:
                    try:
                        delay = max(float(retry_after), delay)
                    except ValueError:
                        pass
                logger.warning(f"{model}: {last_error}; attempt {attempt + 1}, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                continue
            if status == 400 and reasoning_cfg is not None:
                # Some providers reject reasoning controls (e.g. thinking cannot be disabled):
                # fall back to provider defaults rather than failing the job.
                logger.warning(f"{model}: rejects reasoning={reasoning}; using provider default from now on")
                _NO_REASONING_CONTROL.add(model)
                reasoning_cfg = None
                continue
            logger.error(f"{model}: non-retryable {last_error}")
            raise LLMError(f"{model}: {last_error}") from e

        except (APITimeoutError, APIConnectionError, _TransientError) as e:
            last_error = f"{type(e).__name__}: {e}"
            delay = _backoff(attempt)
            logger.warning(f"{model}: {last_error}; attempt {attempt + 1}, retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
            continue

    raise LLMError(f"{model}: failed after {config.MAX_RETRIES + 1} attempts ({last_error})")


class _TransientError(Exception):
    pass
