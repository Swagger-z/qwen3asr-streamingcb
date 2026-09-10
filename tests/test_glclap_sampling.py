"""Tests for deterministic raw-size-proportional corpus schedules."""

from __future__ import annotations

import unittest
from collections import Counter

from asr.contextual.glclap_sampling import proportional_epoch_indices


class ProportionalSamplingTests(unittest.TestCase):
    def test_compute_matched_schedule_uses_exact_raw_size_quotas(self) -> None:
        groups = {"large": list(range(6)), "small": [6, 7]}
        values = proportional_epoch_indices(groups, seed=42, epoch=0, samples_per_epoch=4)
        counts = Counter("large" if value < 6 else "small" for value in values)
        self.assertEqual(counts, {"large": 3, "small": 1})
        self.assertEqual(len(values), len(set(values)))

    def test_full_epoch_covers_every_record_once_and_is_reproducible(self) -> None:
        groups = {"a": [0, 1, 2], "b": [3, 4]}
        first = proportional_epoch_indices(groups, seed=7, epoch=3)
        second = proportional_epoch_indices(groups, seed=7, epoch=3)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(5)))

    def test_single_corpus_full_epoch_preserves_legacy_shuffle(self) -> None:
        import random

        expected = list(range(8))
        random.Random(44).shuffle(expected)
        actual = proportional_epoch_indices({"legacy": list(range(8))}, seed=42, epoch=2)
        self.assertEqual(actual, expected)

    def test_oversampling_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            proportional_epoch_indices({"a": [0]}, seed=1, epoch=0, samples_per_epoch=2)


if __name__ == "__main__":
    unittest.main()
