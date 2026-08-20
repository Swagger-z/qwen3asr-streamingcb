"""Opt-in 100k catalog memory/time smoke test for Linux experiment hosts."""

from __future__ import annotations

import os
import time
import unittest

from asr.contextual.catalog import HotwordCatalog
from asr.contextual.retriever import AhoCorasickRetriever
from asr.contextual.types import HotwordEntry


@unittest.skipUnless(os.environ.get("RUN_CONTEXTUAL_STRESS") == "1", "set RUN_CONTEXTUAL_STRESS=1")
class CatalogStressTests(unittest.TestCase):
    def test_one_hundred_thousand_entries(self) -> None:
        started = time.perf_counter()
        entries = [HotwordEntry(id=f"hw-{index}", text=f"entity{index:06d}") for index in range(100_000)]
        catalog = HotwordCatalog(entries)
        matcher = AhoCorasickRetriever(catalog.text_trie, top_k_candidates=20)
        update = matcher.consume(tuple("entity099999"), 0, 0, 0)
        elapsed = time.perf_counter() - started
        self.assertTrue(any(item.hotword_id == "hw-99999" for item in update.completed))
        self.assertLess(elapsed, 60.0)


if __name__ == "__main__":
    unittest.main()
