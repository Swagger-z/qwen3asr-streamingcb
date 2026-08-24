"""Regression tests for GLCLAP spoken-term negative filtering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from asr.contextual.glclap_data import (
    batch_negative_exclusions,
    SharedNegativeSampler,
    sample_shared_negatives,
)
from asr.contextual.glclap_runtime import load_negative_vocabulary


class NegativeFilteringTests(unittest.TestCase):
    def test_all_spoken_substrings_are_excluded_from_negative_sampling(self) -> None:
        exclusions = batch_negative_exclusions(
            ["今天 去参观人工智能岛"],
            min_chars=2,
            max_chars=4,
        )
        self.assertIn("参观", exclusions)
        self.assertIn("人工智能", exclusions)
        self.assertIn("今天去参观人工智能岛", exclusions)
        negatives = sample_shared_negatives(
            ["参观", "人工智能", "大学", "公司"],
            exclusions,
            count=2,
            strict=True,
        )
        self.assertEqual(set(negatives), {"大学", "公司"})

    def test_precanonicalized_sampler_matches_reference(self) -> None:
        vocabulary = ["公司", "大学", "公司", "", "人工智能", "参观"]
        expected = sample_shared_negatives(
            vocabulary,
            {"参观"},
            count=3,
            seed=7,
            epoch=2,
            step=5,
        )
        actual = SharedNegativeSampler(vocabulary).sample(
            {"参观"},
            count=3,
            seed=7,
            epoch=2,
            step=5,
        )
        self.assertEqual(actual, expected)

    def test_invalid_exclusion_range_fails_early(self) -> None:
        with self.assertRaises(ValueError):
            batch_negative_exclusions(["人工智能"], min_chars=4, max_chars=2)

    def test_raw_word_frequency_text_drops_frequency_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "word_freq.txt"
            path.write_text("就是 10998\n你 41277\n参观 16\n", encoding="utf-8")
            values = load_negative_vocabulary(path)
        self.assertEqual(values, ["你", "参观", "就是"])


if __name__ == "__main__":
    unittest.main()
