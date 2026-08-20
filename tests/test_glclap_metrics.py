"""Standalone GLCLAP retrieval metric tests."""

from __future__ import annotations

import unittest

from asr.eval.glclap_metrics import evaluate_glclap_records, paired_bootstrap_mean_ci


def record(utt_id: str, boundary: str, hit_ids: list[str], gold: str):
    hits = [
        {"hotword_id": item, "score": 1.0 / rank, "rank": rank, "peak_frame": 0, "variant": item}
        for rank, item in enumerate(hit_ids, 1)
    ]
    batch = {"chunk_id": 0, "hits": hits, "timings_ms": {"search_ms": 2.0}}
    return {
        "utt_id": utt_id,
        "target_hotword_ids": [gold],
        "boundary_group": boundary,
        "batches": [batch],
        "final_batch": batch,
        "rtf": 0.1,
        "offline_final_exact_match": True,
    }


class GLCLAPMetricTests(unittest.TestCase):
    def test_quality_boundary_latency_and_parity(self) -> None:
        metrics = evaluate_glclap_records(
            [record("c", "Center", ["gold"], "gold"), record("x", "Cross-50", ["bad"], "gold")]
        )
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["boundary_penalty"], 1.0)
        self.assertEqual(metrics["search_ms_p95"], 2.0)
        self.assertEqual(metrics["offline_final_exact_match_rate"], 1.0)

    def test_bootstrap_is_deterministic(self) -> None:
        self.assertEqual(
            paired_bootstrap_mean_ci([0.0, 1.0, 1.0], samples=100, seed=7),
            paired_bootstrap_mean_ci([0.0, 1.0, 1.0], samples=100, seed=7),
        )


if __name__ == "__main__":
    unittest.main()
