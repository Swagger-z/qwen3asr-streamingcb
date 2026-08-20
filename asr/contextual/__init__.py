"""Contextual biasing primitives for streaming LLM-ASR.

The package is intentionally independent from the Qwen and vLLM runtimes so
that retrieval, commitment, and tracing behavior can be tested on CPU.
"""

from .candidates import CandidateManager
from .catalog import HotwordCatalog
from .commit import CommitController
from .glclap import (
    AccumulatedAudioRetrievalSession,
    AudioEncoding,
    HotwordEmbeddingIndex,
    RetrievalBatch,
    RetrievalHit,
)
from .glclap_model import (
    GLCLAPAdapters,
    GLCLAPRetrieverModel,
    QwenGLCLAPEncoder,
)
from .retriever import AhoCorasickRetriever, CTCPosteriorTrieRetriever
from .session import ContextualSessionConfig, ContextualStreamingSession
from .token_bias import SparseTokenBias
from .types import (
    CandidateEvidence,
    CandidateSnapshot,
    CandidateStatus,
    HotwordEntry,
    ProbeChunk,
    RetrievalUpdate,
    StreamingResult,
    RetrieverState,
)

__all__ = [
    "AhoCorasickRetriever",
    "AccumulatedAudioRetrievalSession",
    "AudioEncoding",
    "CTCPosteriorTrieRetriever",
    "CandidateEvidence",
    "CandidateManager",
    "CandidateSnapshot",
    "CandidateStatus",
    "CommitController",
    "ContextualSessionConfig",
    "ContextualStreamingSession",
    "HotwordCatalog",
    "HotwordEmbeddingIndex",
    "HotwordEntry",
    "GLCLAPAdapters",
    "GLCLAPRetrieverModel",
    "ProbeChunk",
    "QwenGLCLAPEncoder",
    "RetrievalBatch",
    "RetrievalHit",
    "RetrievalUpdate",
    "StreamingResult",
    "RetrieverState",
    "SparseTokenBias",
]
