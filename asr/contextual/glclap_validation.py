"""Validation scheduling and rank metrics for GLCLAP training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ValidationSchedule:
    """Select either epoch-end or optimizer-step validation."""

    strategy: str
    steps: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ValidationSchedule":
        """Build and validate a schedule from the ``evaluation`` config."""

        strategy = str(config.get("strategy", "epoch")).strip().lower()
        if strategy not in {"epoch", "steps"}:
            raise ValueError("evaluation.strategy must be 'epoch' or 'steps'")
        steps = int(config.get("steps", 0))
        if strategy == "steps" and steps <= 0:
            raise ValueError("evaluation.steps must be positive when strategy='steps'")
        return cls(strategy=strategy, steps=steps)

    def after_optimizer_step(self, global_step: int) -> bool:
        """Return whether an optimizer update should trigger validation."""

        return self.strategy == "steps" and global_step > 0 and global_step % self.steps == 0

    def after_epoch(self) -> bool:
        """Return whether every completed epoch should trigger validation."""

        return self.strategy == "epoch"


def retrieval_rank_metrics(
    scores: Any,
    positive_mask: Any,
    *,
    ks: Sequence[int] = (1, 5, 10, 20, 50),
) -> dict[str, float]:
    """Compute best-positive rank, Recall@K, and MRR for a score matrix.

    ``scores`` and ``positive_mask`` must have shape ``[B, K]``. If several
    catalog variants are positive, the best ranked positive is used.
    """

    values = np.asarray(scores)
    positives = np.asarray(positive_mask, dtype=bool)
    if values.ndim != 2 or positives.shape != values.shape:
        raise ValueError("scores and positive_mask must have equal [B, K] shape")
    if values.shape[0] == 0:
        raise ValueError("validation batch cannot be empty")
    if not positives.any(axis=1).all():
        raise ValueError("every validation row must contain at least one positive")
    positive_scores = np.where(positives, values, -np.inf).max(axis=1)
    ranks = 1 + (values > positive_scores[:, None]).sum(axis=1)
    metrics = {f"recall_at_{int(k)}": float(np.mean(ranks <= int(k))) for k in ks}
    metrics["mrr"] = float(np.mean(1.0 / ranks))
    metrics["count"] = float(values.shape[0])
    return metrics
