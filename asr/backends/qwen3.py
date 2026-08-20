"""Pinned, lazy Qwen3-ASR vLLM streaming adapter."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from asr.contextual.catalog import TokenizerLike


@dataclass(frozen=True)
class BackendDecodeResult:
    """One accumulated-audio refresh result."""

    raw_text: str
    text: str
    language: str = ""


class StreamingLLMBackend(Protocol):
    """Backend contract consumed by ``ContextualStreamingSession``."""

    tokenizer: TokenizerLike

    def decode_accumulated(
        self,
        pcm16k: Sequence[float],
        context: str,
        prefix_text: str,
        logits_processor: Any | None = None,
    ) -> BackendDecodeResult:
        """Decode all audio seen so far using a stable text prefix."""


class QwenVLLMBackend:
    """Adapter around ``qwen-asr==0.0.6`` and ``vllm==0.14.0``.

    Imports are delayed until construction so dependency-free unit tests and
    evaluation tooling remain usable on non-CUDA machines.
    """

    EXPECTED_QWEN_ASR_VERSION = "0.0.6"
    EXPECTED_VLLM_VERSION = "0.14.0"
    EXPECTED_TRANSFORMERS_VERSION = "4.57.6"

    def __init__(self, asr_model: Any, language: str | None = None) -> None:
        if getattr(asr_model, "backend", None) != "vllm":
            raise ValueError("QwenVLLMBackend requires Qwen3ASRModel.LLM(...)")
        self.asr_model = asr_model
        self.language = language
        self.tokenizer = asr_model.processor.tokenizer
        self._parse_output = self._load_parser()

    @classmethod
    def from_pretrained(
        cls,
        model: str = "Qwen/Qwen3-ASR-0.6B",
        language: str | None = None,
        **vllm_kwargs: Any,
    ) -> "QwenVLLMBackend":
        """Load the pinned official vLLM backend."""

        try:
            import importlib.metadata as metadata
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:  # pragma: no cover - optional runtime.
            raise ImportError("install the 'qwen' extra on Linux CUDA") from exc
        cls._require_version(metadata, "qwen-asr", cls.EXPECTED_QWEN_ASR_VERSION)
        cls._require_version(metadata, "vllm", cls.EXPECTED_VLLM_VERSION)
        cls._require_version(metadata, "transformers", cls.EXPECTED_TRANSFORMERS_VERSION)
        asr = Qwen3ASRModel.LLM(model=model, **vllm_kwargs)
        return cls(asr, language=language)

    @staticmethod
    def _require_version(metadata: Any, package: str, expected: str) -> None:
        actual = metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package}=={expected} required, found {actual}")

    @staticmethod
    def _load_parser() -> Any:
        try:
            from qwen_asr.inference.utils import parse_asr_output

            return parse_asr_output
        except ImportError:
            return None

    def _prompt_raw(self, context: str) -> str:
        state = self.asr_model.init_streaming_state(
            context=context,
            language=self.language,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=2.0,
        )
        return state.prompt_raw

    def decode_accumulated(
        self,
        pcm16k: Sequence[float],
        context: str,
        prefix_text: str,
        logits_processor: Any | None = None,
    ) -> BackendDecodeResult:
        """Re-feed accumulated audio using an externally selected prefix."""

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Qwen backend requires NumPy") from exc
        prompt = self._prompt_raw(context) + prefix_text
        params = copy.deepcopy(self.asr_model.sampling_params)
        if logits_processor is not None:
            if not hasattr(params, "logits_processors"):
                raise RuntimeError("pinned vLLM logits processor contract is unavailable")
            current = list(getattr(params, "logits_processors", None) or [])
            current.append(logits_processor)
            params.logits_processors = current
        request = {
            "prompt": prompt,
            "multi_modal_data": {"audio": [np.asarray(pcm16k, dtype=np.float32)]},
        }
        outputs = self.asr_model.model.generate([request], sampling_params=params, use_tqdm=False)
        generated = outputs[0].outputs[0].text
        raw_text = prefix_text + generated
        if self._parse_output is None:
            return BackendDecodeResult(raw_text=raw_text, text=raw_text)
        language, text = self._parse_output(raw_text, user_language=self.language)
        return BackendDecodeResult(raw_text=raw_text, text=text, language=language)
