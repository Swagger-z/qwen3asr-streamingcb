"""Dependency-free test doubles for contextual streaming tests."""

from __future__ import annotations

from collections import deque
from typing import Any, Sequence

from asr.backends.qwen3 import BackendDecodeResult


class CharTokenizer:
    """One-Unicode-codepoint-per-token tokenizer."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(int(token)) for token in token_ids)


class ScriptedBackend:
    """Return deterministic full hypotheses while recording request inputs."""

    def __init__(self, hypotheses: Sequence[str]) -> None:
        self.tokenizer = CharTokenizer()
        self._hypotheses = deque(hypotheses)
        self.calls: list[dict[str, Any]] = []

    def decode_accumulated(
        self,
        pcm16k: Sequence[float],
        context: str,
        prefix_text: str,
        logits_processor: Any | None = None,
    ) -> BackendDecodeResult:
        if not self._hypotheses:
            raise AssertionError("no scripted hypothesis remains")
        hypothesis = self._hypotheses.popleft()
        if not hypothesis.startswith(prefix_text):
            raise AssertionError(f"scripted hypothesis {hypothesis!r} does not preserve prefix {prefix_text!r}")
        self.calls.append(
            {
                "audio_samples": len(pcm16k),
                "context": context,
                "prefix_text": prefix_text,
                "has_logits_processor": logits_processor is not None,
            }
        )
        return BackendDecodeResult(raw_text=hypothesis, text=hypothesis, language="test")
