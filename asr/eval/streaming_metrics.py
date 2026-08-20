"""Aggregation helpers for streaming latency and bounded-state diagnostics."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Mapping, Sequence


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for finite values."""

    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return 0.0
    position = (len(clean) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def streaming_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate optional per-utterance streaming measurements.

    Inputs may expose either scalar summary fields or raw arrays. Missing fields
    are omitted so an evaluation cannot silently report invented zero latency.
    """

    arrays = {
        "chunk_latency_ms": ("chunk_processing_ms", "chunk_latencies_ms"),
        "cpu_retrieval_ms": ("cpu_retrieval_ms",),
        "active_states": ("active_state_counts", "active_states"),
        "prompt_tokens": ("prompt_token_counts", "prompt_tokens"),
    }
    result: dict[str, float] = {}
    for output_name, input_names in arrays.items():
        values: list[float] = []
        for record in records:
            for input_name in input_names:
                raw = record.get(input_name)
                if isinstance(raw, (list, tuple)):
                    values.extend(float(item) for item in raw)
                    break
        if values:
            result[f"{output_name}_mean"] = mean(values)
            result[f"{output_name}_p50"] = percentile(values, 0.50)
            result[f"{output_name}_p95"] = percentile(values, 0.95)
            result[f"{output_name}_max"] = max(values)

    scalar_names = (
        "ttft_ms",
        "stable_final_latency_ms",
        "hotword_stable_latency_ms",
        "rtf",
        "revision_rate",
        "gpu_memory_mb",
        "redecoded_tokens_per_minute",
    )
    for name in scalar_names:
        values = [float(record[name]) for record in records if record.get(name) is not None]
        if values:
            result[f"{name}_mean"] = mean(values)
            result[f"{name}_p50"] = percentile(values, 0.50)
            result[f"{name}_p95"] = percentile(values, 0.95)
    return result
