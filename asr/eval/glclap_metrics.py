"""Metrics for standalone GLCLAP hotword retrieval experiments."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _hits(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if "final_batch" in record:
        return list(record["final_batch"].get("hits", []))
    return list(record.get("final_hits", []))


def _gold(record: Mapping[str, Any]) -> set[str]:
    values = record.get("target_hotword_ids", record.get("hotword_ids", ()))
    if isinstance(values, str):
        values = [values]
    return {str(value) for value in values}


def record_recall_at_k(record: Mapping[str, Any], k: int) -> float:
    """Return target-level Recall@K for one utterance."""

    gold = _gold(record)
    if not gold:
        return 0.0
    predicted = {str(hit["hotword_id"]) for hit in _hits(record)[:k]}
    return len(gold & predicted) / len(gold)


def paired_bootstrap_mean_ci(
    differences: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a deterministic percentile CI over paired mean differences."""

    if not differences:
        raise ValueError("differences must be non-empty")
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_glclap_records(
    records: Iterable[Mapping[str, Any]],
    *,
    ks: Sequence[int] = (1, 5, 10, 20, 50),
) -> dict[str, float | int]:
    """Aggregate retrieval quality, boundary, and latency measurements."""

    items = list(records)
    metrics: dict[str, float | int] = {"utterances": len(items)}
    evaluable = [record for record in items if _gold(record)]
    metrics["evaluable_utterances"] = len(evaluable)
    metrics["target_entity_count"] = sum(len(_gold(record)) for record in evaluable)
    for k in ks:
        recalls = [record_recall_at_k(record, int(k)) for record in evaluable]
        precisions = []
        for record in evaluable:
            predicted = [str(hit["hotword_id"]) for hit in _hits(record)[: int(k)]]
            true_count = len(set(predicted) & _gold(record))
            precisions.append(true_count / max(1, len(predicted)))
        recall = float(np.mean(recalls)) if recalls else 0.0
        precision = float(np.mean(precisions)) if precisions else 0.0
        metrics[f"recall_at_{k}"] = recall
        # Hit@K is any-positive success; Recall@K measures entity coverage.
        metrics[f"hit_at_{k}"] = float(np.mean([value > 0 for value in recalls])) if recalls else 0.0
        metrics[f"precision_at_{k}"] = precision
        metrics[f"f1_at_{k}"] = 2 * recall * precision / max(1e-12, recall + precision)
        metrics[f"false_alarm_rate_at_{k}"] = 1.0 - precision

    reciprocal_ranks = []
    first_chunks = []
    for record in evaluable:
        gold = _gold(record)
        rank = next(
            (index for index, hit in enumerate(_hits(record), 1) if str(hit["hotword_id"]) in gold),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        first = None
        for batch in record.get("batches", []):
            if any(str(hit["hotword_id"]) in gold for hit in batch.get("hits", [])[:50]):
                first = int(batch.get("chunk_id", 0))
                break
        if first is not None:
            first_chunks.append(first)
    metrics["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    metrics["first_top50_chunk_mean"] = float(np.mean(first_chunks)) if first_chunks else -1.0

    center = [record_recall_at_k(record, 50) for record in evaluable if _boundary(record) == "center"]
    cross = [record_recall_at_k(record, 50) for record in evaluable if _boundary(record) == "cross"]
    center_recall = float(np.mean(center)) if center else 0.0
    cross_recall = float(np.mean(cross)) if cross else 0.0
    metrics["recall_center_at_50"] = center_recall
    metrics["recall_cross_at_50"] = cross_recall
    metrics["center_utterances"] = len(center)
    metrics["cross_utterances"] = len(cross)
    metrics["boundary_penalty"] = center_recall - cross_recall

    timings: dict[str, list[float]] = {}
    for record in items:
        for batch in record.get("batches", []):
            for name, value in batch.get("timings_ms", {}).items():
                timings.setdefault(str(name), []).append(float(value))
    for name, values in timings.items():
        metrics[f"{name}_p50"] = float(np.percentile(values, 50))
        metrics[f"{name}_p95"] = float(np.percentile(values, 95))

    rtfs = [float(record["rtf"]) for record in items if record.get("rtf") is not None]
    peaks = [float(record["peak_memory_mb"]) for record in items if record.get("peak_memory_mb") is not None]
    metrics["rtf_mean"] = float(np.mean(rtfs)) if rtfs else 0.0
    metrics["peak_memory_mb_max"] = max(peaks) if peaks else 0.0
    exact = [bool(record["offline_final_exact_match"]) for record in items if "offline_final_exact_match" in record]
    if exact:
        metrics["offline_final_exact_match_rate"] = float(np.mean(exact))
    return metrics


def _boundary(record: Mapping[str, Any]) -> str:
    label = str(record.get("boundary_group", "")).strip().casefold()
    if label.startswith("center"):
        return "center"
    if label.startswith("cross"):
        return "cross"
    return "other"
