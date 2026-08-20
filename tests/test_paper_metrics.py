"""Biased/unbiased error partition tests."""

from __future__ import annotations

import unittest

from asr.eval.paper_metrics import evaluate_paper_records


class PaperMetricTests(unittest.TestCase):
    def test_bwer_and_uwer_are_reported_separately(self) -> None:
        records = [
            {
                "reference": "hello rareword now",
                "hypothesis": "hello wrong now",
                "hotwords": ["rareword"],
                "boundary_group": "cross-50",
            }
        ]
        metrics = evaluate_paper_records(records)
        self.assertEqual(metrics["bwer"], 1.0)
        self.assertEqual(metrics["uwer"], 0.0)


if __name__ == "__main__":
    unittest.main()
