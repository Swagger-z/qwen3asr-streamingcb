"""Tests for language-aware GLCLAP positives, negatives, and local losses."""

from __future__ import annotations

import unittest

import numpy as np

from asr.contextual.glclap_data import (
    SharedNegativeSampler,
    batch_negative_exclusions,
    deterministic_local_positive,
    transcript_positive_mask,
)


class MultilingualDataTests(unittest.TestCase):
    def test_english_matching_obeys_word_boundaries_case_and_punctuation(self) -> None:
        mask = transcript_positive_mask(
            ["Taylor Swift's ART show, in New York."],
            ["taylor", "SWIFT'S", "art", "new york", "york", "tar", "cart"],
            language="en",
        )
        self.assertEqual(mask.tolist(), [[True, True, True, True, True, False, False]])

    def test_english_positive_is_a_reproducible_whole_word_span(self) -> None:
        first = deterministic_local_positive(
            "ONE two three four five",
            utt_id="u1",
            epoch=2,
            language="en",
            min_words=2,
            max_words=3,
        )
        second = deterministic_local_positive(
            "ONE two three four five",
            utt_id="u1",
            epoch=2,
            language="en",
            min_words=2,
            max_words=3,
        )
        self.assertEqual(first, second)
        self.assertIn(first, "ONE two three four five")
        self.assertIn(len(first.split()), (2, 3))

    def test_english_exclusions_and_sampler_are_case_insensitive(self) -> None:
        exclusions = batch_negative_exclusions(
            ["NEW York is lovely"], language="en", min_words=1, max_words=2
        )
        self.assertIn("new york", exclusions)
        sampler = SharedNegativeSampler(
            ["New York", "BOSTON", "Seattle"], language="en"
        )
        sampled = sampler.sample({"NEW YORK"}, 2, strict=True)
        self.assertEqual(set(sampled), {"BOSTON", "Seattle"})


class MultilingualLossTests(unittest.TestCase):
    def test_language_local_losses_cover_mixed_batch(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional")
        from asr.contextual.glclap_loss import multilingual_glclap_loss

        torch.manual_seed(1)
        audio = torch.randn(4, 3, 5, requires_grad=True)
        transcript = torch.randn(4, 5, requires_grad=True)
        hotwords = {"zh": torch.randn(3, 5), "en": torch.randn(3, 5)}
        global_mask = torch.eye(4, dtype=torch.bool)
        local_masks = {
            "zh": torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool),
            "en": torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool),
        }
        output = multilingual_glclap_loss(
            audio,
            transcript,
            hotwords,
            global_mask,
            local_masks,
            {"zh": [0, 2], "en": [1, 3]},
            temperature=torch.tensor(0.07),
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.local_logits["zh"].shape, (2, 3))
        self.assertEqual(output.local_logits["en"].shape, (2, 3))
        output.loss.backward()
        self.assertIsNotNone(audio.grad)


if __name__ == "__main__":
    unittest.main()
