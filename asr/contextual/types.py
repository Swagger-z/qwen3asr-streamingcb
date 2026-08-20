"""Shared public types for contextual streaming ASR."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Hashable, Mapping, Sequence


@dataclass(frozen=True)
class HotwordEntry:
    """One catalog entry.

    Attributes:
        id: Stable identifier used by session filters and traces.
        text: Canonical display spelling.
        aliases: Alternative spellings accepted by retrieval and token biasing.
        language: Language tag such as ``zh``, ``en``, or ``mixed``.
        weight: Non-negative application prior used only for candidate ranking.
        metadata: JSON-compatible auxiliary fields. ``pronunciations`` may contain
            strings or lists of phoneme symbols.
    """

    id: str
    text: str
    aliases: tuple[str, ...] = ()
    language: str = ""
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("hotword id must be non-empty")
        if not self.text.strip():
            raise ValueError("hotword text must be non-empty")
        if self.weight < 0:
            raise ValueError("hotword weight must be non-negative")


class CandidateStatus(str, Enum):
    """Lifecycle state for a per-session hotword candidate."""

    ACTIVE = "active"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class CandidateEvidence:
    """Evidence emitted by a text or acoustic retriever."""

    hotword_id: str
    confidence: float
    start_frame: int | None = None
    anchor_token: int | None = None
    complete: bool = False


@dataclass(frozen=True)
class CandidateSnapshot:
    """Serializable view of one managed candidate."""

    hotword_id: str
    status: CandidateStatus
    confidence: float
    anchor_token: int
    start_frame: int | None
    first_chunk: int
    last_update_chunk: int


@dataclass(frozen=True)
class RetrievalUpdate:
    """Batch of candidate observations produced for one chunk."""

    active: tuple[CandidateEvidence, ...] = ()
    completed: tuple[CandidateEvidence, ...] = ()
    rejected_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeChunk:
    """Phoneme posterior frames with an absolute frame offset.

    ``log_probs`` has shape ``[T, P]`` and ``symbols`` maps its vocabulary
    columns to the hashable symbols stored in the pronunciation trie.
    """

    log_probs: Sequence[Sequence[float]]
    frame_offset: int
    blank_id: int
    symbols: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        if self.frame_offset < 0:
            raise ValueError("frame_offset must be non-negative")
        if not 0 <= self.blank_id < len(self.symbols):
            raise ValueError("blank_id must index symbols")
        width = len(self.symbols)
        if any(len(row) != width for row in self.log_probs):
            raise ValueError("every log-probability frame must match symbols")


@dataclass(frozen=True)
class RetrieverState:
    """Serializable view of one cross-chunk CTC trie path.

    ``last_token_id`` and ``ended_blank`` expose the CTC repeat state in
    addition to the paper-facing fields needed for traces and diagnostics.
    """

    trie_node: int
    ctc_forward_score: float
    candidate_id: str
    start_audio_frame: int
    start_text_token: int
    confidence: float
    last_update_chunk: int
    last_token_id: int
    ended_blank: bool

    def __post_init__(self) -> None:
        if self.trie_node < 0:
            raise ValueError("trie_node must be non-negative")
        if self.start_audio_frame < 0 or self.start_text_token < 0:
            raise ValueError("state anchors must be non-negative")


@dataclass(frozen=True)
class StreamingResult:
    """User-visible result returned after a streaming update."""

    partial_text: str
    stable_text: str
    committed_delta: str
    active_candidates: tuple[CandidateSnapshot, ...]
    confirmed_candidates: tuple[CandidateSnapshot, ...]
    rollback_start_token: int
    chunk_id: int
    processing_ms: float
    is_final: bool = False
    debug: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""

        data = asdict(self)
        for key in ("active_candidates", "confirmed_candidates"):
            for item in data[key]:
                status = item.get("status")
                if isinstance(status, Enum):
                    item["status"] = status.value
        return data
