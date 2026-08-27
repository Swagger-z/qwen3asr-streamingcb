"""Tests for deterministic GLCLAP validation scheduling and ranks."""

from __future__ import annotations

import unittest

import numpy as np

from asr.contextual.glclap_validation import ValidationSchedule, retrieval_rank_metrics


class ValidationScheduleTests(unittest.TestCase):
    def test_epoch_strategy_only_runs_after_epoch(self) -> None:
        schedule = ValidationSchedule.from_config({"strategy": "epoch", "steps": 10})
        self.assertTrue(schedule.after_epoch())
        self.assertFalse(schedule.after_optimizer_step(10))

    def test_step_strategy_counts_optimizer_updates(self) -> None:
        schedule = ValidationSchedule.from_config({"strategy": "steps", "steps": 3})
        self.assertFalse(schedule.after_epoch())
        self.assertFalse(schedule.after_optimizer_step(2))
        self.assertTrue(schedule.after_optimizer_step(3))
        self.assertTrue(schedule.after_optimizer_step(6))

    def test_invalid_strategy_or_interval_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strategy"):
            ValidationSchedule.from_config({"strategy": "batch"})
        with self.assertRaisesRegex(ValueError, "positive"):
            ValidationSchedule.from_config({"strategy": "steps", "steps": 0})


class RetrievalRankMetricTests(unittest.TestCase):
    def test_best_positive_rank_supports_multi_positive_rows(self) -> None:
        scores = np.asarray([[0.9, 0.1, 0.2], [0.7, 0.6, 0.5]], dtype=np.float32)
        positives = np.asarray([[True, False, True], [False, False, True]])
        metrics = retrieval_rank_metrics(scores, positives, ks=(1, 2, 3))
        self.assertEqual(metrics["count"], 2.0)
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["recall_at_2"], 0.5)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], (1.0 + 1.0 / 3.0) / 2.0)

    def test_missing_positive_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            retrieval_rank_metrics(np.zeros((1, 2)), np.zeros((1, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()
