"""Regression tests for GLCLAP spoken-term negative filtering."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from asr.contextual.glclap_data import (
    batch_negative_exclusions,
    equality_positive_mask,
    SharedNegativeSampler,
    sample_shared_negatives,
    transcript_positive_mask,
)
from asr.contextual.glclap_runtime import load_negative_vocabulary


class NegativeFilteringTests(unittest.TestCase):
    def test_other_rows_positive_is_also_positive_when_spoken(self) -> None:
        transcripts = ["我在北京工作", "北京欢迎你"]
        candidates = ["工作", "北京", "上海"]
        mask = transcript_positive_mask(transcripts, candidates)
        self.assertEqual(mask.tolist(), [[True, True, False], [False, True, False]])
        # Transposing for text->audio must also keep both Beijing audios positive.
        self.assertEqual(mask.T[1].tolist(), [True, True])

    def test_matching_uses_sampling_normalization_and_no_fuzzy_matches(self) -> None:
        mask = transcript_positive_mask(
            ["我在北 京 工 作", "在ＡＢＣ上班"],
            ["北京", "工 作", "ＡＢＣ", "ABC", "背景", "", "   "],
        )
        self.assertEqual(mask.tolist(), [
            [True, True, False, False, False, False, False],
            [False, False, True, True, False, False, False],
        ])

    def test_overlapping_terms_duplicates_and_own_positives_are_preserved(self) -> None:
        mask = transcript_positive_mask(
            ["北京大学欢迎你", "上海大学欢迎你"],
            ["北京大学", "北京", "北京", "上海大学", "背景大学"],
        )
        self.assertEqual(mask.tolist(), [
            [True, True, True, False, False],
            [False, False, False, True, False],
        ])

    def test_nonoverlapping_batch_matches_previous_labels(self) -> None:
        positives = ["北京", "上海"]
        candidates = [*positives, "广州"]
        mask = transcript_positive_mask(["北京欢迎你", "上海欢迎你"], candidates)
        self.assertEqual(mask.tolist(), equality_positive_mask(positives, candidates).tolist())

    def test_48_audio_batch_with_4095_random_negatives(self) -> None:
        transcripts = [f"北京研究所第{index}组" for index in range(48)]
        candidates = ["北京", *(f"第{index}组" for index in range(1, 48)),
                      *(f"未出现词{index}" for index in range(4095))]
        original_candidates = list(candidates)
        mask = transcript_positive_mask(transcripts, candidates)
        self.assertEqual(mask.shape, (48, 48 + 4095))
        self.assertTrue(mask[:, 0].all())
        self.assertFalse(mask[:, 48:].any())
        self.assertTrue(all(mask[index, index] for index in range(48)))
        self.assertEqual(candidates, original_candidates)

    def test_empty_candidates_preserve_global_only_shape(self) -> None:
        self.assertEqual(transcript_positive_mask(["北京", "上海"], []).shape, (2, 0))
        self.assertEqual(transcript_positive_mask([], ["北京"]).shape, (0, 1))

    def test_training_uses_transcripts_but_validation_keeps_gold_labels(self) -> None:
        source = Path(__file__).resolve().parents[1] / "scripts/train_glclap_retriever.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for function_name, builder in (
            ("main", "transcript_positive_mask"),
            ("_evaluate", "membership_positive_mask"),
        ):
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == function_name)
            calls = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == builder
            ]
            self.assertEqual(len(calls), 1)
            if function_name == "_evaluate":
                self.assertEqual(calls[0].args[0].id, "positive_groups")
            else:
                self.assertEqual(calls[0].args[0].id, "group_transcripts")
                self.assertTrue(
                    any(keyword.arg == "language" for keyword in calls[0].keywords)
                )

    def test_all_spoken_substrings_are_excluded_from_negative_sampling(self) -> None:
        exclusions = batch_negative_exclusions(
            ["今天 去参观人工智能岛"],
            min_chars=2,
            max_chars=4,
        )
        self.assertIn("参观", exclusions)
        self.assertIn("人工智能", exclusions)
        self.assertIn("今天去参观人工智能岛", exclusions)
        negatives = sample_shared_negatives(
            ["参观", "人工智能", "大学", "公司"],
            exclusions,
            count=2,
            strict=True,
        )
        self.assertEqual(set(negatives), {"大学", "公司"})

    def test_precanonicalized_sampler_matches_reference(self) -> None:
        vocabulary = ["公司", "大学", "公司", "", "人工智能", "参观"]
        expected = sample_shared_negatives(
            vocabulary,
            {"参观"},
            count=3,
            seed=7,
            epoch=2,
            step=5,
        )
        actual = SharedNegativeSampler(vocabulary).sample(
            {"参观"},
            count=3,
            seed=7,
            epoch=2,
            step=5,
        )
        self.assertEqual(actual, expected)

    def test_invalid_exclusion_range_fails_early(self) -> None:
        with self.assertRaises(ValueError):
            batch_negative_exclusions(["人工智能"], min_chars=4, max_chars=2)

    def test_raw_word_frequency_text_drops_frequency_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "word_freq.txt"
            path.write_text("就是 10998\n你 41277\n参观 16\n", encoding="utf-8")
            values = load_negative_vocabulary(path)
        self.assertEqual(values, ["你", "参观", "就是"])


if __name__ == "__main__":
    unittest.main()
