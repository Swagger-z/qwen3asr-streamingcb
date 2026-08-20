"""Per-session candidate lifecycle and prompt selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .catalog import HotwordCatalog
from .types import CandidateEvidence, CandidateSnapshot, CandidateStatus, RetrievalUpdate


@dataclass
class _Candidate:
    hotword_id: str
    status: CandidateStatus
    confidence: float
    anchor_token: int
    start_frame: int | None
    first_chunk: int
    last_update_chunk: int

    def snapshot(self) -> CandidateSnapshot:
        return CandidateSnapshot(
            hotword_id=self.hotword_id,
            status=self.status,
            confidence=self.confidence,
            anchor_token=self.anchor_token,
            start_frame=self.start_frame,
            first_chunk=self.first_chunk,
            last_update_chunk=self.last_update_chunk,
        )


class CandidateManager:
    """Maintain active/confirmed/rejected/expired contextual candidates."""

    def __init__(
        self,
        catalog: HotwordCatalog,
        enabled_ids: Iterable[str] | None = None,
        enter_threshold: float = 0.55,
        exit_threshold: float = 0.35,
        active_ttl_chunks: int = 3,
        confirmed_ttl_chunks: int = 2,
        retrieval_top_k: int = 20,
        prompt_top_k: int = 5,
    ) -> None:
        if not 0 <= exit_threshold <= enter_threshold <= 1:
            raise ValueError("require 0 <= exit_threshold <= enter_threshold <= 1")
        self.catalog = catalog
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.active_ttl_chunks = active_ttl_chunks
        self.confirmed_ttl_chunks = confirmed_ttl_chunks
        self.retrieval_top_k = retrieval_top_k
        self.prompt_top_k = prompt_top_k
        self.enabled_ids = set(enabled_ids) if enabled_ids is not None else set(catalog.entries)
        unknown = self.enabled_ids - set(catalog.entries)
        if unknown:
            raise KeyError(f"unknown hotword ids: {sorted(unknown)}")
        self._candidates: dict[str, _Candidate] = {}

    def update_enabled(self, enabled_ids: Iterable[str], chunk_id: int) -> None:
        """Replace the session filter; removed active candidates expire now."""

        new_enabled = set(enabled_ids)
        unknown = new_enabled - set(self.catalog.entries)
        if unknown:
            raise KeyError(f"unknown hotword ids: {sorted(unknown)}")
        removed = self.enabled_ids - new_enabled
        self.enabled_ids = new_enabled
        for hotword_id in removed:
            candidate = self._candidates.get(hotword_id)
            if candidate is not None:
                candidate.status = CandidateStatus.EXPIRED
                candidate.last_update_chunk = chunk_id

    def apply(self, update: RetrievalUpdate, chunk_id: int, default_anchor_token: int) -> None:
        """Apply one retrieval update and advance candidate lifecycles."""

        for evidence in update.active:
            self._observe(evidence, CandidateStatus.ACTIVE, chunk_id, default_anchor_token)
        for evidence in update.completed:
            self._observe(evidence, CandidateStatus.CONFIRMED, chunk_id, default_anchor_token)
        for hotword_id in update.rejected_ids:
            candidate = self._candidates.get(hotword_id)
            if candidate is not None:
                candidate.status = CandidateStatus.REJECTED
                candidate.last_update_chunk = chunk_id
        self.expire(chunk_id)
        self._bound_state()

    def _observe(
        self,
        evidence: CandidateEvidence,
        status: CandidateStatus,
        chunk_id: int,
        default_anchor_token: int,
    ) -> None:
        if evidence.hotword_id not in self.enabled_ids:
            return
        confidence = max(0.0, min(1.0, float(evidence.confidence)))
        existing = self._candidates.get(evidence.hotword_id)
        threshold = self.enter_threshold if existing is None else self.exit_threshold
        if confidence < threshold:
            return
        anchor = default_anchor_token if evidence.anchor_token is None else evidence.anchor_token
        if existing is None:
            self._candidates[evidence.hotword_id] = _Candidate(
                hotword_id=evidence.hotword_id,
                status=status,
                confidence=confidence,
                anchor_token=max(0, int(anchor)),
                start_frame=evidence.start_frame,
                first_chunk=chunk_id,
                last_update_chunk=chunk_id,
            )
            return
        existing.status = status
        existing.confidence = max(existing.confidence, confidence) if status is CandidateStatus.CONFIRMED else confidence
        existing.anchor_token = min(existing.anchor_token, max(0, int(anchor)))
        if evidence.start_frame is not None:
            existing.start_frame = evidence.start_frame if existing.start_frame is None else min(existing.start_frame, evidence.start_frame)
        existing.last_update_chunk = chunk_id

    def expire(self, chunk_id: int) -> None:
        """Expire candidates whose per-status TTL has elapsed."""

        for candidate in self._candidates.values():
            age = chunk_id - candidate.last_update_chunk
            if candidate.status is CandidateStatus.ACTIVE and age >= self.active_ttl_chunks:
                candidate.status = CandidateStatus.EXPIRED
            elif candidate.status is CandidateStatus.CONFIRMED and age >= self.confirmed_ttl_chunks:
                candidate.status = CandidateStatus.EXPIRED

    def _rank(self, candidate: _Candidate) -> tuple[float, str]:
        return (candidate.confidence * self.catalog.weight(candidate.hotword_id), candidate.hotword_id)

    def _bound_state(self) -> None:
        live = [
            candidate
            for candidate in self._candidates.values()
            if candidate.status in (CandidateStatus.ACTIVE, CandidateStatus.CONFIRMED)
        ]
        if len(live) <= self.retrieval_top_k:
            return
        keep = {item.hotword_id for item in sorted(live, key=self._rank, reverse=True)[: self.retrieval_top_k]}
        for candidate in live:
            if candidate.hotword_id not in keep:
                candidate.status = CandidateStatus.EXPIRED

    def holding_candidates(self) -> tuple[CandidateSnapshot, ...]:
        """Return active candidates that are allowed to hold commitment."""

        candidates = [
            candidate
            for candidate in self._candidates.values()
            if candidate.status is CandidateStatus.ACTIVE and candidate.confidence >= self.exit_threshold
        ]
        return tuple(item.snapshot() for item in sorted(candidates, key=self._rank, reverse=True))

    def confirmed_candidates(self) -> tuple[CandidateSnapshot, ...]:
        """Return confirmed candidates ordered by ranking score."""

        candidates = [candidate for candidate in self._candidates.values() if candidate.status is CandidateStatus.CONFIRMED]
        return tuple(item.snapshot() for item in sorted(candidates, key=self._rank, reverse=True))

    def prompt_entries(self) -> tuple[HotwordEntry, ...]:
        """Select a small hysteretic dynamic context list."""

        live = [
            candidate
            for candidate in self._candidates.values()
            if candidate.status in (CandidateStatus.CONFIRMED, CandidateStatus.ACTIVE)
        ]
        chosen = sorted(live, key=self._rank, reverse=True)[: self.prompt_top_k]
        return tuple(self.catalog.entries[item.hotword_id] for item in chosen)

    def confidence_by_id(self) -> dict[str, float]:
        """Return live candidate confidence for logit gating."""

        return {
            item.hotword_id: item.confidence
            for item in self._candidates.values()
            if item.status in (CandidateStatus.ACTIVE, CandidateStatus.CONFIRMED)
        }

    def reset(self) -> None:
        """Remove all per-stream candidate state."""

        self._candidates.clear()


# Imported late only for the public return annotation above.
from .types import HotwordEntry  # noqa: E402
