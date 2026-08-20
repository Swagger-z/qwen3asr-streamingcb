"""End-to-end dependency-free contextual session tests."""

from __future__ import annotations

import unittest

from asr.contextual.catalog import HotwordCatalog
from asr.contextual.session import ContextualSessionConfig, ContextualStreamingSession
from asr.contextual.types import HotwordEntry
from tests.fakes import ScriptedBackend


class ContextualSessionTests(unittest.TestCase):
    def test_long_active_hotword_rolls_back_before_fixed_five(self) -> None:
        backend = ScriptedBackend(
            [
                "prefixabcde",
                "prefixabcdefg",
                "prefixabcdefghi",
            ]
        )
        catalog = HotwordCatalog([HotwordEntry(id="long", text="abcdefghi")])
        config = ContextualSessionConfig(
            sample_rate=10,
            chunk_size_ms=100,
            base_rollback_tokens=5,
            max_rollback_tokens=32,
            unfixed_chunk_num=2,
            enable_token_bias=True,
            enable_transcript_probe=True,
        )
        session = ContextualStreamingSession(backend, catalog, config)
        first = session.step([0.0])
        self.assertEqual(first.active_candidates[0].anchor_token, 6)
        session.step([0.0])
        third = session.step([0.0])
        self.assertEqual(backend.calls[2]["prefix_text"], "prefix")
        self.assertLess(backend.calls[2]["audio_samples"], 100)
        self.assertIn("long", third.debug["prompt_hotword_ids"])
        final = session.finish()
        self.assertTrue(final.is_final)
        self.assertEqual(final.stable_text, "prefixabcdefghi")

    def test_update_applies_on_next_chunk_and_reset_is_complete(self) -> None:
        backend = ScriptedBackend(["hel", "hello", "x"])
        catalog = HotwordCatalog(
            [HotwordEntry(id="hello", text="hello"), HotwordEntry(id="world", text="world")]
        )
        config = ContextualSessionConfig(sample_rate=10, chunk_size_ms=100, unfixed_chunk_num=0)
        session = ContextualStreamingSession(backend, catalog, config, enabled_ids=["hello"])
        session.step([0.0])
        self.assertTrue(session.candidates.holding_candidates())
        session.update_hotwords(enabled_ids=["world"])
        session.step([0.0])
        self.assertFalse(any(item.hotword_id == "hello" for item in session.candidates.holding_candidates()))
        session.reset()
        self.assertEqual(session.chunk_id, 0)
        result = session.step([0.0])
        self.assertEqual(result.chunk_id, 0)


if __name__ == "__main__":
    unittest.main()
