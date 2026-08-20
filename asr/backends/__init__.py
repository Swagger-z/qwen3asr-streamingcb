"""Runtime adapters for external LLM-ASR engines."""

from .qwen3 import BackendDecodeResult, QwenVLLMBackend, StreamingLLMBackend

__all__ = ["BackendDecodeResult", "QwenVLLMBackend", "StreamingLLMBackend"]
