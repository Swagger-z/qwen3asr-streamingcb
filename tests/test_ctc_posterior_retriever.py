"""CTC blank/repeat and cross-chunk token-passing tests."""

from __future__ import annotations

import math
import unittest

from asr.contextual.catalog import HotwordCatalog
from asr.contextual.retriever import CTCPosteriorTrieRetriever
from asr.contextual.types import HotwordEntry, ProbeChunk


def lp(blank: float, a: float, b: float) -> list[float]:
    return [math.log(blank), math.log(a), math.log(b)]


class CTCPosteriorRetrieverTests(unittest.TestCase):
    def test_repeat_blank_and_cross_chunk_completion(self) -> None:
        catalog = HotwordCatalog(
            [HotwordEntry(id="ab", text="ab", metadata={"pronunciations": [["a", "b"]]})]
        )
        retriever = CTCPosteriorTrieRetriever(
            catalog.pronunciation_trie,
            max_active_states=8,
            top_tokens_per_frame=3,
        )
        first = ProbeChunk(
            log_probs=[lp(0.01, 0.98, 0.01), lp(0.01, 0.98, 0.01), lp(0.98, 0.01, 0.01)],
            frame_offset=0,
            blank_id=0,
            symbols=("<blank>", "a", "b"),
        )
        update1 = retriever.consume(first, anchor_token=4, chunk_id=0)
        self.assertTrue(any(item.hotword_id == "ab" for item in update1.active))
        second = ProbeChunk(
            log_probs=[lp(0.01, 0.01, 0.98)],
            frame_offset=3,
            blank_id=0,
            symbols=("<blank>", "a", "b"),
        )
        update2 = retriever.consume(second, anchor_token=99, chunk_id=1)
        match = next(item for item in update2.completed if item.hotword_id == "ab")
        self.assertEqual(match.anchor_token, 4)
        self.assertLessEqual(retriever.active_state_count, 8)


if __name__ == "__main__":
    unittest.main()
