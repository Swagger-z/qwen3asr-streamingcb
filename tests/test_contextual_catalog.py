"""Catalog and exact cross-chunk matcher tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from asr.contextual.catalog import HotwordCatalog
from asr.contextual.retriever import AhoCorasickRetriever
from asr.contextual.types import HotwordEntry


class CatalogTests(unittest.TestCase):
    def test_versioned_jsonl_and_multi_path_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hotwords.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "catalog_version": "paper-v1",
                        "id": "hw1",
                        "text": "Alex",
                        "aliases": ["Alexander"],
                        "language": "en",
                        "metadata": {"pronunciations": [["AE", "L", "AH", "K", "S"]]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            catalog = HotwordCatalog.from_jsonl(path)
        self.assertEqual(catalog.version, "paper-v1")
        self.assertEqual(catalog.entries["hw1"].aliases, ("Alexander",))
        paths = [path for path, hotword_id in catalog.pronunciation_trie.iter_paths() if hotword_id == "hw1"]
        self.assertIn(("AE", "L", "AH", "K", "S"), paths)

    def test_stateful_matcher_completes_across_chunks(self) -> None:
        entry = HotwordEntry(
            id="island",
            text="人工智能岛",
            metadata={"pronunciations": [["ren", "gong", "zhi", "neng", "dao"]]},
        )
        catalog = HotwordCatalog([entry])
        matcher = AhoCorasickRetriever(catalog.pronunciation_trie, stateful=True)
        first = matcher.consume(("ren", "gong", "zhi"), frame_offset=0, anchor_token=7, chunk_id=0)
        self.assertEqual(first.active[0].hotword_id, "island")
        second = matcher.consume(("neng", "dao"), frame_offset=3, anchor_token=20, chunk_id=1)
        self.assertEqual(second.completed[0].hotword_id, "island")
        self.assertEqual(second.completed[0].anchor_token, 20)

    def test_stateless_matcher_drops_partial_path(self) -> None:
        entry = HotwordEntry(id="x", text="abc", metadata={"pronunciations": [["a", "b", "c"]]})
        catalog = HotwordCatalog([entry])
        matcher = AhoCorasickRetriever(catalog.pronunciation_trie, stateful=False)
        self.assertFalse(matcher.consume(("a", "b"), 0, 0, 0).completed)
        self.assertFalse(matcher.consume(("c",), 2, 2, 1).completed)


if __name__ == "__main__":
    unittest.main()
