"""Optional pinned Qwen/vLLM integration contract checks."""

from __future__ import annotations

import importlib.util
import os
import unittest

from asr.backends import QwenVLLMBackend


HAS_RUNTIME = importlib.util.find_spec("qwen_asr") is not None and importlib.util.find_spec("vllm") is not None


@unittest.skipUnless(HAS_RUNTIME and os.environ.get("QWEN_MODEL_PATH"), "requires pinned Qwen runtime and model")
class QwenContractTests(unittest.TestCase):
    def test_adapter_loads_pinned_runtime(self) -> None:
        backend = QwenVLLMBackend.from_pretrained(
            model=os.environ["QWEN_MODEL_PATH"],
            gpu_memory_utilization=0.5,
            max_new_tokens=8,
        )
        self.assertIsNotNone(backend.tokenizer)


if __name__ == "__main__":
    unittest.main()
