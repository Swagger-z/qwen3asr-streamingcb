"""Contextual and streaming evaluation metrics."""

from __future__ import annotations

import math
import random
import statistics
import unicodedata
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


def normalize_metric_text(text: str) -> str:
    """Normalize case, width, and whitespace for metric computation."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def metric_tokens(text: str, character: bool | None = None) -> list[str]:
    """Tokenize by whitespace or characters for character-dominant text."""

    normalized = normalize_metric_text(text)
    if character is None:
        character = " " not in normalized
    return [char for char in normalized if not char.isspace()] if character else normalized.split()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return Levenshtein distance using linear memory."""

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, 1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, character: bool | None = None) -> float:
    """Return WER/CER-like edit rate."""

    ref = metric_tokens(reference, character)
    hyp = metric_tokens(hypothesis, character)
    return edit_distance(ref, hyp) / max(1, len(ref))


def keyword_counts(reference: str, hypothesis: str, hotwords: Iterable[str]) -> dict[str, float]:
    """Count exact normalized keyword occurrences and derive P/R/F1/FAR."""

    ref = normalize_metric_text(reference)
    hyp = normalize_metric_text(hypothesis)
    ref_count = 0
    hyp_count = 0
    true_positive = 0
    for hotword in hotwords:
        key = normalize_metric_text(hotword)
        if not key:
            continue
        expected = ref.count(key)
        predicted = hyp.count(key)
        ref_count += expected
        hyp_count += predicted
        true_positive += min(expected, predicted)
    false_positive = max(0, hyp_count - true_positive)
    precision = true_positive / max(1, hyp_count)
    recall = true_positive / max(1, ref_count)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "keyword_ref": float(ref_count),
        "keyword_hyp": float(hyp_count),
        "keyword_tp": float(true_positive),
        "keyword_fp": float(false_positive),
        "keyword_precision": precision,
        "keyword_recall": recall,
        "keyword_f1": f1,
        "keyword_far": false_positive / max(1, len(tuple(hotwords))),
    }


def evaluate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate transcript, hotword, boundary, and latency metrics."""

    if not records:
        raise ValueError("at least one record is required")
    total_word_errors = 0
    total_words = 0
    total_char_errors = 0
    total_chars = 0
    keyword_totals: Counter[str] = Counter()
    recall_by_group: dict[str, list[float]] = {}
    latencies: list[float] = []
    revisions: list[float] = []
    rtfs: list[float] = []
    for record in records:
        reference = str(record["reference"])
        hypothesis = str(record["hypothesis"])
        ref_words = metric_tokens(reference, character=False)
        hyp_words = metric_tokens(hypothesis, character=False)
        ref_chars = metric_tokens(reference, character=True)
        hyp_chars = metric_tokens(hypothesis, character=True)
        total_word_errors += edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)
        total_char_errors += edit_distance(ref_chars, hyp_chars)
        total_chars += len(ref_chars)
        counts = keyword_counts(reference, hypothesis, record.get("hotwords", ()))
        for key in ("keyword_ref", "keyword_hyp", "keyword_tp", "keyword_fp"):
            keyword_totals[key] += counts[key]
        group = str(record.get("boundary_group", "unspecified"))
        recall_by_group.setdefault(group, []).append(counts["keyword_recall"])
        if record.get("hotword_stable_latency_ms") is not None:
            latencies.append(float(record["hotword_stable_latency_ms"]))
        if record.get("revision_rate") is not None:
            revisions.append(float(record["revision_rate"]))
        if record.get("rtf") is not None:
            rtfs.append(float(record["rtf"]))
    precision = keyword_totals["keyword_tp"] / max(1, keyword_totals["keyword_hyp"])
    recall = keyword_totals["keyword_tp"] / max(1, keyword_totals["keyword_ref"])
    result = {
        "wer": total_word_errors / max(1, total_words),
        "cer": total_char_errors / max(1, total_chars),
        "keyword_precision": precision,
        "keyword_recall": recall,
        "keyword_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "keyword_far": keyword_totals["keyword_fp"] / len(records),
    }
    center = recall_by_group.get("center")
    cross = [value for group, values in recall_by_group.items() if group.startswith("cross") for value in values]
    if center and cross:
        result["boundary_penalty"] = statistics.fmean(center) - statistics.fmean(cross)
    if latencies:
        result["hotword_stable_latency_ms_mean"] = statistics.fmean(latencies)
    if revisions:
        result["revision_rate_mean"] = statistics.fmean(revisions)
    if rtfs:
        result["rtf_mean"] = statistics.fmean(rtfs)
    return result


def paired_bootstrap_ci(
    differences: Sequence[float],
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a paired percentile bootstrap confidence interval."""

    if not differences:
        raise ValueError("differences must be non-empty")
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(statistics.fmean(rng.choice(differences) for _ in differences))
    means.sort()
    alpha = (1 - confidence) / 2
    lower = means[max(0, math.floor(alpha * samples))]
    upper = means[min(samples - 1, math.ceil((1 - alpha) * samples) - 1)]
    return lower, upper
