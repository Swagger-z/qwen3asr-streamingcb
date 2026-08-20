"""Tests for streaming experiment metric aggregation."""

from __future__ import annotations

import unittest

from asr.eval.streaming_metrics import percentile, streaming_aggregates


class StreamingMetricTests(unittest.TestCase):
    def test_percentile_and_optional_fields(self) -> None:
        records = [
            {"chunk_processing_ms": [1.0, 3.0], "ttft_ms": 10.0},
            {"chunk_processing_ms": [5.0], "ttft_ms": 20.0},
        ]
        result = streaming_aggregates(records)
        self.assertEqual(percentile([1.0, 3.0, 5.0], 0.5), 3.0)
        self.assertAlmostEqual(result["chunk_latency_ms_p95"], 4.8)
        self.assertEqual(result["ttft_ms_mean"], 15.0)
        self.assertNotIn("gpu_memory_mb_mean", result)


if __name__ == "__main__":
    unittest.main()
