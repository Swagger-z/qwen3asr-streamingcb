"""Dependency-light GLCLAP index and accumulated-session tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from asr.contextual.glclap import (
    AccumulatedAudioRetrievalSession,
    AudioEncoding,
    HotwordEmbeddingIndex,
)
from asr.contextual.glclap_data import (
    deterministic_local_positive,
    equality_positive_mask,
    sample_shared_negatives,
)


class _LengthEncoder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def encode_pcm(self, pcm16k):
        self.calls.append(len(pcm16k))
        return AudioEncoding(
            np.asarray([[1.0, float(len(pcm16k)) / 16000]], dtype=np.float32),
            {"aut_ms": 1.0, "projector_ms": 0.2, "adapter_ms": 0.1},
        )


class GLCLAPIndexTests(unittest.TestCase):
    def test_alias_aggregation_keeps_best_variant_and_peak(self) -> None:
        index = HotwordEmbeddingIndex(
            np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
            ["same", "same", "other"],
            ["canonical", "alias", "other"],
            block_size=1,
        )
        audio = np.asarray([[0.1, 0.9], [1.0, 0.0]], dtype=np.float32)
        result = index.search(audio, top_k=2)
        self.assertEqual(result.hits[0].hotword_id, "same")
        self.assertEqual(result.hits[0].variant, "canonical")
        self.assertEqual(result.hits[0].peak_frame, 1)
        self.assertEqual(len(result.hits), 2)

    def test_blockwise_matches_dense_partition_and_roundtrip(self) -> None:
        rng = np.random.default_rng(42)
        keys = rng.normal(size=(17, 8)).astype(np.float32)
        ids = [f"id-{index // 2}" for index in range(17)]
        variants = [f"v-{index}" for index in range(17)]
        audio = rng.normal(size=(11, 8)).astype(np.float32)
        blockwise = HotwordEmbeddingIndex(keys, ids, variants, block_size=3)
        dense = HotwordEmbeddingIndex(keys, ids, variants, block_size=1000)
        blockwise_hits = blockwise.search(audio, 50).hits
        dense_hits = dense.search(audio, 50).hits
        self.assertEqual(
            [(hit.hotword_id, hit.rank, hit.peak_frame, hit.variant) for hit in blockwise_hits],
            [(hit.hotword_id, hit.rank, hit.peak_frame, hit.variant) for hit in dense_hits],
        )
        for left, right in zip(blockwise_hits, dense_hits):
            self.assertAlmostEqual(left.score, right.score, places=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.bin"
            blockwise.save(path)
            restored = HotwordEmbeddingIndex.load(path)
        self.assertEqual(blockwise.search(audio, 50).hits, restored.search(audio, 50).hits)

    def test_empty_catalog_is_a_noop(self) -> None:
        index = HotwordEmbeddingIndex(np.empty((0, 4), dtype=np.float32), [], [])
        result = index.search(np.ones((3, 4), dtype=np.float32))
        self.assertEqual(result.hits, ())
        self.assertEqual(result.frame_count, 3)


class AccumulatedSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = HotwordEmbeddingIndex(
            np.asarray([[1.0, 0.0]], dtype=np.float32), ["one"], ["一"]
        )

    def test_step_tail_finish_reset_and_index_guard(self) -> None:
        encoder = _LengthEncoder()
        session = AccumulatedAudioRetrievalSession(
            encoder, self.index, chunk_size_sec=2.0, sample_rate=16000
        )
        self.assertEqual(session.step([0.0] * 16000), ())
        first = session.step([0.0] * 16000)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].accumulated_audio_sec, 2.0)
        self.assertEqual(encoder.calls, [32000])
        with self.assertRaises(RuntimeError):
            session.set_index(self.index)
        self.assertEqual(session.step([0.0] * 8000), ())
        final = session.finish()
        self.assertTrue(final.is_final)
        self.assertEqual(final.accumulated_audio_sec, 2.5)
        self.assertEqual(encoder.calls, [32000, 40000])
        with self.assertRaises(RuntimeError):
            session.step([0.0])
        session.reset()
        session.set_index(self.index)
        self.assertEqual(session.step([]), ())

    def test_multiple_completed_chunks_and_session_isolation(self) -> None:
        left_encoder = _LengthEncoder()
        right_encoder = _LengthEncoder()
        left = AccumulatedAudioRetrievalSession(left_encoder, self.index, chunk_size_sec=2.0)
        right = AccumulatedAudioRetrievalSession(right_encoder, self.index, chunk_size_sec=2.0)
        batches = left.step([0.0] * (5 * 16000))
        self.assertEqual([item.accumulated_audio_sec for item in batches], [2.0, 4.0])
        self.assertEqual(right_encoder.calls, [])
        right.step([0.0] * (2 * 16000))
        self.assertEqual(left_encoder.calls, [32000, 64000])
        self.assertEqual(right_encoder.calls, [32000])


class GLCLAPDataTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_excludes_positives(self) -> None:
        first = deterministic_local_positive("欢迎来到人工智能岛", utt_id="u1", epoch=3)
        second = deterministic_local_positive("欢迎来到人工智能岛", utt_id="u1", epoch=3)
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 2)
        negatives = sample_shared_negatives(
            ["甲", "乙", "丙", first], [first], count=2, strict=True
        )
        self.assertNotIn(first, negatives)

    def test_multi_positive_mask_marks_duplicates(self) -> None:
        mask = equality_positive_mask(["甲", "甲", "乙"], ["甲", "乙", "甲"])
        self.assertEqual(mask.tolist(), [[True, False, True], [True, False, True], [False, True, False]])


if __name__ == "__main__":
    unittest.main()
