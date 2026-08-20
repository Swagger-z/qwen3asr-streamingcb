"""Dependency-free version pin contract tests."""

from __future__ import annotations

import unittest

from asr.backends import QwenVLLMBackend


class QwenPinTests(unittest.TestCase):
    def test_runtime_versions_are_locked(self) -> None:
        self.assertEqual(QwenVLLMBackend.EXPECTED_QWEN_ASR_VERSION, "0.0.6")
        self.assertEqual(QwenVLLMBackend.EXPECTED_VLLM_VERSION, "0.14.0")
        self.assertEqual(QwenVLLMBackend.EXPECTED_TRANSFORMERS_VERSION, "4.57.6")


if __name__ == "__main__":
    unittest.main()
