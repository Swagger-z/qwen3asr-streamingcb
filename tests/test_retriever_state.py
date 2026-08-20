"""Public cross-chunk retriever-state contract tests."""

from __future__ import annotations

import math
import unittest

from asr.contextual import HotwordCatalog, HotwordEntry, ProbeChunk, RetrieverState
from asr.contextual.retriever import CTCPosteriorTrieRetriever


class RetrieverStateTests(unittest.TestCase):
    def test_state_exposes_acoustic_and_text_anchors(self) -> None:
        catalog = HotwordCatalog(
            [HotwordEntry(id="ab", text="ab", metadata={"pronunciations": [["a", "b"]]})]
        )
        retriever = CTCPosteriorTrieRetriever(catalog.pronunciation_trie, max_active_states=8)
        retriever.consume(
            ProbeChunk(
                log_probs=[[math.log(0.01), math.log(0.98), math.log(0.01)]],
                frame_offset=12,
                blank_id=0,
                symbols=("<blank>", "a", "b"),
            ),
            anchor_token=7,
            chunk_id=3,
        )
        state = next(item for item in retriever.states if item.candidate_id == "ab")
        self.assertIsInstance(state, RetrieverState)
        self.assertEqual(state.start_audio_frame, 12)
        self.assertEqual(state.start_text_token, 7)
        self.assertEqual(state.last_update_chunk, 3)
        self.assertGreater(state.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
