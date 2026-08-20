"""BWER/UWER partitioning layered on the core contextual metrics."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .streaming_metrics import streaming_aggregates
from .contextual_metrics import evaluate_records, metric_tokens, normalize_metric_text


def _alignment(reference: Sequence[str], hypothesis: Sequence[str]) -> list[tuple[str, str | None, str | None]]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    table = [[0] * cols for _ in range(rows)]
    for index in range(rows):
        table[index][0] = index
    for index in range(cols):
        table[0][index] = index
    for i in range(1, rows):
        for j in range(1, cols):
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
            )
    operations = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            operation = "match" if reference[i - 1] == hypothesis[j - 1] else "substitute"
            operations.append((operation, reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            operations.append(("delete", reference[i - 1], None))
            i -= 1
        else:
            operations.append(("insert", None, hypothesis[j - 1]))
            j -= 1
    operations.reverse()
    return operations


def partitioned_error_counts(
    reference: str,
    hypothesis: str,
    hotwords: Iterable[str],
    character: bool | None = None,
) -> dict[str, int]:
    """Partition edit errors by whether the affected unit belongs to a hotword."""

    if character is None:
        character = " " not in normalize_metric_text(reference)
    ref = metric_tokens(reference, character)
    hyp = metric_tokens(hypothesis, character)
    bias_units = {token for hotword in hotwords for token in metric_tokens(hotword, character)}
    counts = {"biased_ref": 0, "biased_errors": 0, "unbiased_ref": 0, "unbiased_errors": 0}
    for operation, ref_token, hyp_token in _alignment(ref, hyp):
        token = ref_token if ref_token is not None else hyp_token
        biased = token in bias_units
        if ref_token is not None:
            counts["biased_ref" if biased else "unbiased_ref"] += 1
        if operation != "match":
            counts["biased_errors" if biased else "unbiased_errors"] += 1
    return counts


def evaluate_paper_records(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return core, BWER/UWER, and available streaming metrics."""

    result = evaluate_records(records)
    totals = {"biased_ref": 0, "biased_errors": 0, "unbiased_ref": 0, "unbiased_errors": 0}
    for record in records:
        counts = partitioned_error_counts(
            str(record["reference"]),
            str(record["hypothesis"]),
            record.get("hotwords", ()),
        )
        for key, value in counts.items():
            totals[key] += value
    result["bwer"] = totals["biased_errors"] / max(1, totals["biased_ref"])
    result["uwer"] = totals["unbiased_errors"] / max(1, totals["unbiased_ref"])
    return result

    result.update(streaming_aggregates(records))
