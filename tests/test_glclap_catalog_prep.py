"""Dependency-free tests for GLCLAP word-frequency catalog preparation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from asr.contextual.catalog_prep import (
    WordFrequencyEntry,
    build_evaluation_catalog,
    build_negative_catalog,
    merge_word_frequency_sources,
    parse_word_frequency_line,
    select_frequency_stratified,
    validate_manifest_targets,
)


def entry(text: str, frequency: int) -> WordFrequencyEntry:
    return WordFrequencyEntry(text=text, source_frequencies=(("mixed", frequency),))


class WordFrequencyParsingTests(unittest.TestCase):
    def test_parser_uses_final_column_as_frequency(self) -> None:
        self.assertEqual(
            parse_word_frequency_line("就是 10998", source="hkust", line_number=1),
            ("就是", 10998),
        )
        with self.assertRaisesRegex(ValueError, "hkust:2"):
            parse_word_frequency_line("参观 not-a-count", source="hkust", line_number=2)

    def test_merge_filters_single_char_and_preserves_source_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hkust = root / "hkust.txt"
            magicdata = root / "magicdata.txt"
            hkust.write_text("就是 10\n你 20\n参观 5\n就是 2\n", encoding="utf-8")
            magicdata.write_text("就是 3\n那边 4\nenglish 8\n", encoding="utf-8")
            entries, reports = merge_word_frequency_sources(
                [("hkust", hkust), ("magicdata", magicdata)]
            )
        by_text = {item.text: item for item in entries}
        self.assertEqual(set(by_text), {"就是", "参观", "那边"})
        self.assertEqual(by_text["就是"].total_frequency, 15)
        self.assertEqual(dict(by_text["就是"].source_frequencies), {"hkust": 12, "magicdata": 3})
        self.assertEqual([item.unique_terms for item in reports], [2, 2])


class CatalogConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        terms = [
            "就是",
            "参观",
            "那边",
            "上面",
            "今天",
            "明天",
            "欢迎",
            "来到",
            "人工",
            "智能",
            "大学",
            "公司",
        ]
        self.entries = [entry(text, 10_000 - index * 100) for index, text in enumerate(terms)]

    def test_frequency_sampling_is_exact_deterministic_and_excludes_terms(self) -> None:
        left = select_frequency_stratified(
            self.entries,
            limit=6,
            excluded_terms={"就是", "参观"},
            seed=42,
            bucket_count=3,
        )
        right = select_frequency_stratified(
            self.entries,
            limit=6,
            excluded_terms={"就是", "参观"},
            seed=42,
            bucket_count=3,
        )
        self.assertEqual(left, right)
        self.assertEqual(len(left), 6)
        self.assertFalse({"就是", "参观"} & {item.text for item in left})

    def test_training_catalog_has_stable_ids_and_provenance(self) -> None:
        records = build_negative_catalog(self.entries, size=4, seed=7)
        self.assertEqual(len(records), 4)
        self.assertTrue(all(record["id"].startswith("neg-") for record in records))
        self.assertTrue(all(record["metadata"]["role"] == "train_negative" for record in records))
        self.assertTrue(all("source_frequencies" in record["metadata"] for record in records))

    def test_evaluation_catalog_keeps_targets_and_removes_spoken_distractors(self) -> None:
        targets = [
            {
                "id": "ne-1",
                "text": "人工智能岛",
                "aliases": ["智能岛"],
                "metadata": {"entity_type": "poi"},
            }
        ]
        records = build_evaluation_catalog(
            targets,
            self.entries,
            size=5,
            evaluation_transcripts=["今天去参观那边的人工智能岛"],
            seed=3,
        )
        self.assertEqual(len(records), 5)
        self.assertEqual(records[0]["id"], "ne-1")
        self.assertEqual(records[0]["metadata"]["role"], "target")
        distractor_texts = {record["text"] for record in records[1:]}
        self.assertFalse({"今天", "参观", "那边", "人工", "智能"} & distractor_texts)

    def test_manifest_target_validation_fails_on_unknown_id(self) -> None:
        targets = [{"id": "known", "text": "人工智能岛"}]
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_manifest_targets(
                [{"utt_id": "u1", "target_hotword_ids": ["missing"]}],
                targets,
            )


if __name__ == "__main__":
    unittest.main()
