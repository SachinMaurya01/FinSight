"""Pydantic request/response models for the FinSight API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

QueueStatus = Literal["completed", "pending_review", "degraded", "abstained"]


class AskRequest(BaseModel):
    """Strict request body for ``POST /v1/ask``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    ticker: str | None = Field(default=None, pattern=r"^[A-Z.\-]{1,10}$")
    fiscal_year: int | None = Field(default=None, ge=2000, le=2100)
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)


class AskResponse(BaseModel):
    """Response returned by ``POST /v1/ask`` and ``POST /v1/reviews/{thread_id}``."""

    thread_id: str
    status: QueueStatus
    answer: str | None = None
    sources: list[str] = Field(default_factory=list)
    provider: str | None = None
    tier: str | None = None
    verification: dict[str, Any] | None = None
    reason: str | None = None
    caveat: str | None = None


class ReviewRequest(BaseModel):
    """Human review decision for a paused thread."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    note: str | None = Field(default=None, max_length=2000)


class PendingReview(BaseModel):
    """A review pause waiting for a human decision."""

    thread_id: str
    query: str | None = None
    answer: str | None = None
    sources: list[str] = Field(default_factory=list)
    reason: str | None = None
    created_at: float | None = None


class PendingReviewsResponse(BaseModel):
    """Collection wrapper for pending reviews."""

    pending: list[PendingReview] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Liveness response."""

    status: Literal["ok"]
    version: str


class ReadinessCheck(BaseModel):
    """Individual dependency readiness state."""

    ok: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Readiness response; ``status`` is ``degraded`` when any check fails."""

    status: Literal["ok", "degraded"]
    checks: dict[str, ReadinessCheck]
