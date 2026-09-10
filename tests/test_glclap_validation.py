"""Tests for deterministic GLCLAP validation scheduling and ranks."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from asr.contextual.glclap_validation import ValidationSchedule, retrieval_rank_metrics
from scripts.train_glclap_retriever import ValidationSet, _aggregate_validation_metrics


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


class MultilingualValidationAggregationTests(unittest.TestCase):
    def test_english_datasets_are_averaged_before_equal_language_macro(self) -> None:
        datasets = (
            ValidationSet("zh-dev", "zh", Path("zh"), (), "a"),
            ValidationSet("en-clean", "en", Path("clean"), (), "b"),
            ValidationSet("en-other", "en", Path("other"), (), "c"),
        )
        def metrics(recall: float, count: int) -> dict[str, float | int]:
            values = {
                name: recall
                for name in (
                    "loss", "global_loss", "local_loss", "recall_at_1",
                    "recall_at_5", "recall_at_10", "recall_at_20",
                    "recall_at_50", "mrr",
                )
            }
            return {**values, "num_examples": count, "num_batches": 1}
        result = _aggregate_validation_metrics(
            datasets,
            {
                "zh-dev": metrics(1.0, 100),
                "en-clean": metrics(0.0, 10),
                "en-other": metrics(0.5, 10),
            },
        )
        self.assertEqual(result["by_language"]["en"]["recall_at_50"], 0.25)
        self.assertEqual(result["macro_language_recall_at_50"], 0.625)


if __name__ == "__main__":
    unittest.main()
