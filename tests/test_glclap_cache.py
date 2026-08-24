"""Tests for persistent frozen Qwen feature caching."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from asr.contextual.glclap_cache import QwenFeatureCache
    from asr.contextual.glclap_model import QwenAudioFeatures


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class QwenFeatureCacheTests(unittest.TestCase):
    def test_post_projector_round_trip_and_namespace_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "audio.wav"
            source.write_bytes(b"pcm")
            cache = QwenFeatureCache(
                root / "cache",
                namespace="model-a",
                feature_kind="post_projector",
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            expected = torch.randn(4, 1024)
            features = QwenAudioFeatures(torch.randn(4, 896), expected)
            self.assertIsNone(cache.get("utt", source))
            cache.put("utt", source, features)
            loaded = cache.get("utt", source)
            self.assertIsNotNone(loaded)
            self.assertTrue(torch.equal(loaded.post_projector, expected))
            self.assertEqual(tuple(loaded.pre_projector.shape), (0, 0))
            isolated = QwenFeatureCache(
                root / "cache",
                namespace="model-b",
                feature_kind="post_projector",
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.assertIsNone(isolated.get("utt", source))


if __name__ == "__main__":
    unittest.main()
