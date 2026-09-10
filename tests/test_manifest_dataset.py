"""Tests for indexed JSONL access and strict multilingual metadata."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from asr.data.manifest_dataset import ManifestDataset


class ManifestDatasetTests(unittest.TestCase):
    def test_random_access_protocol_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            rows = [
                {"key": "a:1", "source_key": "1", "source": "1.wav", "target": "一", "language": "zh", "corpus": "a", "split": "train"},
                {"key": "b:2", "source_key": "2", "source": "2.wav", "target": "TWO", "language": "en", "corpus": "b", "split": "train"},
            ]
            path.write_text(json.dumps(rows[0]) + "\n\n" + json.dumps(rows[1]) + "\n", encoding="utf-8")
            dataset = ManifestDataset(path, strict_multilingual=True)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[1]["target"], "TWO")
            self.assertEqual(dataset[-1]["key"], "b:2")
            self.assertEqual(dataset.protocol_summary()["language_counts"], {"en": 1, "zh": 1})
            self.assertEqual(dataset.indices_by_corpus, {"a": (0,), "b": (1,)})

    def test_strict_schema_rejects_non_namespaced_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(
                json.dumps({"key": "1", "source_key": "1", "source": "1.wav", "target": "x", "language": "en", "corpus": "libri", "split": "train"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "<corpus>:<source_key>"):
                ManifestDataset(path, strict_multilingual=True)


if __name__ == "__main__":
    unittest.main()
