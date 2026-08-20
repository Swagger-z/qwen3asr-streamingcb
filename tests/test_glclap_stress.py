"""Opt-in 10k-key GLCLAP exact-search stress smoke test."""

from __future__ import annotations

import os
import unittest

import numpy as np

from asr.contextual.glclap import HotwordEmbeddingIndex


@unittest.skipUnless(os.environ.get("RUN_GLCLAP_STRESS") == "1", "set RUN_GLCLAP_STRESS=1")
class GLCLAPStressTests(unittest.TestCase):
    def test_ten_thousand_512d_keys(self) -> None:
        rng = np.random.default_rng(42)
        keys = rng.normal(size=(10_000, 512)).astype(np.float32)
        audio = rng.normal(size=(125, 512)).astype(np.float32)
        index = HotwordEmbeddingIndex(
            keys,
            [f"id-{item}" for item in range(10_000)],
            [f"词-{item}" for item in range(10_000)],
            block_size=16384,
        )
        result = index.search(audio, top_k=50)
        self.assertEqual(len(result.hits), 50)
        self.assertEqual(result.frame_count, 125)
        self.assertEqual([hit.rank for hit in result.hits], list(range(1, 51)))


if __name__ == "__main__":
    unittest.main()
