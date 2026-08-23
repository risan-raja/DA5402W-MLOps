"""Prometheus metrics for /predict."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

PREDICT_REQUESTS = Counter("predict_requests_total", "Total /predict requests")
PREDICT_ERRORS = Counter("predict_errors_total", "Failed /predict requests")
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "End-to-end /predict latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
