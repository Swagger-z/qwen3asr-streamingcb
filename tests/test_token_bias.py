"""Sparse token bias path, no-op, and session isolation tests."""

from __future__ import annotations

import unittest

from asr.contextual.catalog import HotwordCatalog
from asr.contextual.token_bias import SparseTokenBias
from asr.contextual.types import HotwordEntry
from tests.fakes import CharTokenizer


class TokenBiasTests(unittest.TestCase):
    def test_multi_path_sparse_children(self) -> None:
        tokenizer = CharTokenizer()
        catalog = HotwordCatalog([HotwordEntry(id="alex", text="Alex", aliases=("ALEX",), weight=1.5)])
        bias = SparseTokenBias(catalog, tokenizer, base_bonus=2.0, max_bonus=4.0)
        root = bias.biases([], {"alex": 0.8})
        self.assertIn(ord("A"), root)
        self.assertIn(ord(" "), root)
        after_a = bias.biases([ord("A")], {"alex": 0.8})
        self.assertEqual(set(after_a), {ord("l"), ord("L")})
        self.assertAlmostEqual(after_a[ord("l")], 2.4)

    def test_disabled_is_exact_noop_and_processors_are_isolated(self) -> None:
        tokenizer = CharTokenizer()
        catalog = HotwordCatalog([HotwordEntry(id="x", text="xy")])
        bias = SparseTokenBias(catalog, tokenizer)
        self.assertEqual(bias.biases([], {}), {})
        first = bias.processor({"x": 1.0})
        second = bias.processor({"x": 0.0})
        logits1 = [0.0] * 256
        logits2 = [0.0] * 256
        first([], logits1)
        second([], logits2)
        self.assertGreater(logits1[ord("x")], 0)
        self.assertEqual(logits2[ord("x")], 0)


if __name__ == "__main__":
    unittest.main()
