"""Pydantic models for API request/response and internal state."""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Job statuses ─────────────────────────────────────────────────────────────
class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


# ── Field result (per-field in the output) ───────────────────────────────────
class FieldResult(BaseModel):
    value: Any = None
    citation: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    page_numbers: Optional[list[int]] = None
    source_locations: Optional[list[str]] = None

    def to_output(self) -> dict:
        """Contract shape: value/citation/confidence always present (value may be null);
        the optional location lists only when we have them."""
        out = {"value": self.value, "citation": self.citation, "confidence": self.confidence}
        if self.page_numbers:
            out["page_numbers"] = self.page_numbers
        if self.source_locations:
            out["source_locations"] = self.source_locations
        return out


# ── Metadata ─────────────────────────────────────────────────────────────────
class ModelBreakdown(BaseModel):
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class ExtractionMetadata(BaseModel):
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    models: dict[str, ModelBreakdown] = Field(default_factory=dict)


# ── Extraction result ────────────────────────────────────────────────────────
class ExtractionResult(BaseModel):
    entries: list[dict[str, FieldResult]]
    metadata: ExtractionMetadata


# ── Job response ─────────────────────────────────────────────────────────────
class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    stage: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[dict] = None


class ErrorResponse(BaseModel):
    code: str
    message: str


# ── Schema (field list) ─────────────────────────────────────────────────────
class SchemaField(BaseModel):
    name: str
    description: str
    primary: Optional[bool] = None


class ExtractionSchema(BaseModel):
    name: str
    description: str = ""
    fields: dict[str, SchemaField]

