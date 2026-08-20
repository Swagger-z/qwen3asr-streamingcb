"""Config fallback, boundary placement, and paper metric tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from asr.config import load_config
from asr.data.boundary_stress import DEFAULT_CONDITIONS, silence_for_condition
from asr.eval.contextual_metrics import evaluate_records, paired_bootstrap_ci


class ConfigMetricTests(unittest.TestCase):
    def test_yaml_fallback_shape_and_override(self) -> None:
        config = load_config(
            Path(__file__).parents[1] / "configs" / "contextual" / "main.yaml",
            ["session.chunk_size_ms=560", "session.enable_token_bias=false"],
        )
        self.assertEqual(config["session"]["chunk_size_ms"], 560)
        self.assertFalse(config["session"]["enable_token_bias"])

    def test_boundary_silence_is_non_negative_and_places_crossing(self) -> None:
        condition = next(item for item in DEFAULT_CONDITIONS if item.name == "cross-50")
        silence = silence_for_condition(0.8, 1.2, 1.0, condition)
        self.assertGreaterEqual(silence, 0)
        boundary_position = 0.8 + silence + 0.2
        self.assertAlmostEqual(boundary_position % 1.0, 0.0)

    def test_boundary_penalty_and_bootstrap(self) -> None:
        records = [
            {"reference": "hello rareword", "hypothesis": "hello rareword", "hotwords": ["rareword"], "boundary_group": "center"},
            {"reference": "hello rareword", "hypothesis": "hello wrong", "hotwords": ["rareword"], "boundary_group": "cross-50"},
        ]
        metrics = evaluate_records(records)
        self.assertEqual(metrics["boundary_penalty"], 1.0)
        low, high = paired_bootstrap_ci([1.0, 1.0, 1.0], samples=100, seed=7)
        self.assertEqual((low, high), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
