"""Pydantic schemas for structured LLM outputs and tool results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TickerPriceResponse(BaseModel):
    """Validated response from get_stock_price tool."""

    ticker: str = Field(description="Uppercase ticker symbol")
    price: float = Field(gt=0, description="Current price in USD")
    currency: str = Field(default="USD")
    as_of: str | None = Field(default=None, description="ISO timestamp of price")

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ticker must not be empty")
        return v.strip().upper()


class RatioResult(BaseModel):
    """Validated result from a financial ratio calculation."""

    ratio_type: Literal["pe_ratio", "gross_margin", "growth_rate"] = Field(
        description="Which ratio was computed"
    )
    value: float = Field(description="Computed ratio value (fraction for margin/growth, absolute for P/E)")
    numerator: float = Field(description="Numerator used")
    denominator: float = Field(description="Denominator used")
    formula: str = Field(description="Human-readable formula")


class ExtractedFinancialMetrics(BaseModel):
    """Structured extraction of financial metrics from filings.

    LLM-extracted; every field is validated. Optional fields may be None when
    not present in the source chunk.
    """

    ticker: str = Field(description="Ticker symbol")
    fiscal_period: str | None = Field(default=None, description="e.g. FY2023")
    revenue: float | None = Field(default=None, ge=0, description="Total net sales / revenue")
    net_income: float | None = Field(default=None, description="Net income (can be negative)")
    eps_diluted: float | None = Field(default=None, description="Diluted EPS")
    gross_margin: float | None = Field(default=None, ge=0, le=1, description="Gross margin fraction 0..1")
    currency: str = Field(default="USD")

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ticker must not be empty")
        return v.strip().upper()
