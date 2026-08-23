"""Pydantic response models for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_name: str
    latency_ms: float
    probabilities: dict[str, float] = Field(default_factory=dict)
