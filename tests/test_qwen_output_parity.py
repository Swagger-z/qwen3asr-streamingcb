"""Optional end-to-end output parity check against official streaming."""

from __future__ import annotations

import importlib.util
import os
import unittest

from asr.audio_io import read_wav_mono_float
from asr.backends import QwenVLLMBackend
from asr.config import load_config
from asr.contextual import ContextualSessionConfig, ContextualStreamingSession, HotwordCatalog


HAS_RUNTIME = importlib.util.find_spec("qwen_asr") is not None and importlib.util.find_spec("vllm") is not None
HAS_FIXTURE = bool(os.environ.get("QWEN_MODEL_PATH") and os.environ.get("QWEN_TEST_WAV"))


@unittest.skipUnless(HAS_RUNTIME and HAS_FIXTURE, "requires pinned Qwen runtime, model, and QWEN_TEST_WAV")
class QwenOutputParityTests(unittest.TestCase):
    def test_no_context_session_matches_official_streaming(self) -> None:
        audio = read_wav_mono_float(os.environ["QWEN_TEST_WAV"])
        sample_rate = 16_000
        backend = QwenVLLMBackend.from_pretrained(
            model=os.environ["QWEN_MODEL_PATH"],
            gpu_memory_utilization=0.5,
        )
        chunk_samples = 2_000 * sample_rate // 1_000

        official = backend.asr_model.init_streaming_state(
            context="",
            chunk_size_sec=2.0,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )
        for start in range(0, len(audio), chunk_samples):
            official = backend.asr_model.streaming_transcribe(
                audio[start : start + chunk_samples],
                official,
                force_streaming=True,
            )
        official = backend.asr_model.finish_streaming_transcribe(official)

        config = load_config("configs/qwen/official_streaming.yaml")
        session_config = ContextualSessionConfig(**config["session"])
        session = ContextualStreamingSession(
            backend,
            HotwordCatalog([], version="empty"),
            session_config,
            static_context="",
        )
        for start in range(0, len(audio), chunk_samples):
            session.step(audio[start : start + chunk_samples])
        result = session.finish()
        self.assertEqual(result.stable_text.strip(), official["text"].strip())


if __name__ == "__main__":
    unittest.main()
