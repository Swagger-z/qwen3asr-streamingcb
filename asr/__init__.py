"""Qwen3-ASR contextual-biasing research package.

Legacy CTC modules keep their original import paths. The primary public
streaming API is exported below.
"""

from .contextual import (
    CandidateManager,
    CommitController,
    ContextualSessionConfig,
    ContextualStreamingSession,
    HotwordCatalog,
    HotwordEntry,
    SparseTokenBias,
    StreamingResult,
)

__all__ = [
    "CandidateManager",
    "CommitController",
    "ContextualSessionConfig",
    "ContextualStreamingSession",
    "HotwordCatalog",
    "HotwordEntry",
    "SparseTokenBias",
    "StreamingResult",
]
