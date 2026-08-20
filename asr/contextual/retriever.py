"""Stateful text/phoneme retrieval across streaming chunks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

from .trie import TokenTrie
from .types import CandidateEvidence, ProbeChunk, RetrievalUpdate, RetrieverState


class AhoCorasickRetriever:
    """Incremental exact matcher over a token or phoneme stream."""

    def __init__(
        self,
        trie: TokenTrie,
        enabled_ids: Iterable[str] | None = None,
        stateful: bool = True,
        top_k_candidates: int = 20,
    ) -> None:
        self.trie = trie
        self.enabled_ids = set(enabled_ids) if enabled_ids is not None else None
        self.stateful = stateful
        self.top_k_candidates = top_k_candidates
        self.node = 0
        self.absolute_position = 0

    def update_enabled(self, enabled_ids: Iterable[str]) -> None:
        """Replace the allowed catalog ID set without replaying history."""

        self.enabled_ids = set(enabled_ids)

    def _enabled(self, hotword_id: str) -> bool:
        return self.enabled_ids is None or hotword_id in self.enabled_ids

    def consume(
        self,
        symbols: Sequence[Hashable],
        frame_offset: int,
        anchor_token: int,
        chunk_id: int,
    ) -> RetrievalUpdate:
        """Consume one new sequence and retain unfinished state when enabled."""

        del chunk_id
        if not self.stateful:
            self.node = 0
        completed: dict[str, CandidateEvidence] = {}
        final_frame = frame_offset - 1
        for offset, symbol in enumerate(symbols):
            final_frame = frame_offset + offset
            self.node = self.trie.step(self.node, symbol)
            for hotword_id in self.trie.output_ids(self.node):
                if not self._enabled(hotword_id):
                    continue
                length = self.trie.min_path_length(hotword_id)
                completed[hotword_id] = CandidateEvidence(
                    hotword_id=hotword_id,
                    confidence=1.0,
                    start_frame=max(0, final_frame - length + 1),
                    anchor_token=anchor_token,
                    complete=True,
                )
        active: list[CandidateEvidence] = []
        if self.stateful and self.node:
            depth = self.trie.nodes[self.node].depth
            start_frame = max(0, final_frame - depth + 1)
            for hotword_id in self.trie.candidate_ids(self.node, limit=self.top_k_candidates):
                if not self._enabled(hotword_id) or hotword_id in completed:
                    continue
                confidence = min(0.99, depth / max(1, self.trie.min_path_length(hotword_id)))
                active.append(
                    CandidateEvidence(
                        hotword_id=hotword_id,
                        confidence=confidence,
                        start_frame=start_frame,
                        anchor_token=anchor_token,
                    )
                )
        if not self.stateful:
            self.node = 0
        self.absolute_position = max(self.absolute_position, frame_offset + len(symbols))
        return RetrievalUpdate(active=tuple(active), completed=tuple(completed.values()))

    def reset(self) -> None:
        """Reset cross-chunk automaton state."""

        self.node = 0
        self.absolute_position = 0


@dataclass(frozen=True)
class _CTCPathKey:
    node: int
    last_token: int
    ended_blank: bool
    start_frame: int
    anchor_token: int


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high = max(left, right)
    return high + math.log(math.exp(left - high) + math.exp(right - high))


class CTCPosteriorTrieRetriever:
    """Stateful pruned CTC token passing over a pronunciation trie.

    Paths retain trie state, CTC repeat/blank state, acoustic start frame, and
    the text anchor captured when the path first opened. Scores for identical
    states are combined in log space and carried across chunk boundaries.
    """

    def __init__(
        self,
        trie: TokenTrie,
        enabled_ids: Iterable[str] | None = None,
        max_active_states: int = 256,
        beam_threshold: float = 12.0,
        top_tokens_per_frame: int = 8,
        top_k_candidates: int = 20,
        min_completion_confidence: float = 0.55,
    ) -> None:
        self.trie = trie
        self.enabled_ids = set(enabled_ids) if enabled_ids is not None else None
        self.max_active_states = max_active_states
        self.beam_threshold = beam_threshold
        self.top_tokens_per_frame = top_tokens_per_frame
        self.top_k_candidates = top_k_candidates
        self.min_completion_confidence = min_completion_confidence
        self._paths: dict[_CTCPathKey, float] = {}

        self._last_chunk_id = -1
        self._last_frame = -1
    @property
    def active_state_count(self) -> int:
        """Return the current bounded token-passing state count."""

        return len(self._paths)

    @property
    def states(self) -> tuple[RetrieverState, ...]:
        """Return bounded public snapshots of retained cross-chunk paths."""

        snapshots: list[RetrieverState] = []
        for key, score in sorted(self._paths.items(), key=lambda item: item[1], reverse=True):
            frames = max(1, self._last_frame - key.start_frame + 1)
            confidence = max(0.0, min(0.999, math.exp(score / frames)))
            candidate_ids = self.trie.candidate_ids(key.node, limit=self.top_k_candidates)
            for candidate_id in candidate_ids:
                if not self._enabled(candidate_id):
                    continue
                snapshots.append(
                    RetrieverState(
                        trie_node=key.node,
                        ctc_forward_score=score,
                        candidate_id=candidate_id,
                        start_audio_frame=key.start_frame,
                        start_text_token=key.anchor_token,
                        confidence=confidence,
                        last_update_chunk=self._last_chunk_id,
                        last_token_id=key.last_token,
                        ended_blank=key.ended_blank,
                    )
                )
                if len(snapshots) >= self.max_active_states:
                    return tuple(snapshots)
        return tuple(snapshots)
    def update_enabled(self, enabled_ids: Iterable[str]) -> None:


        """Replace the allowed ID set for future evidence."""

        self.enabled_ids = set(enabled_ids)

    def _enabled(self, hotword_id: str) -> bool:
        return self.enabled_ids is None or hotword_id in self.enabled_ids

    @staticmethod
    def _merge(target: dict[_CTCPathKey, float], key: _CTCPathKey, score: float) -> None:
        target[key] = _logaddexp(target.get(key, -math.inf), score)

    def consume(self, chunk: ProbeChunk, anchor_token: int, chunk_id: int) -> RetrievalUpdate:
        """Consume new posterior frames and return active/completed evidence."""

        self._last_chunk_id = chunk_id
        completed: dict[str, CandidateEvidence] = {}
        for local_frame, row in enumerate(chunk.log_probs):
            absolute_frame = chunk.frame_offset + local_frame
            top_ids = sorted(range(len(row)), key=lambda index: row[index], reverse=True)[: self.top_tokens_per_frame]
            if chunk.blank_id not in top_ids:
                top_ids.append(chunk.blank_id)
            next_paths: dict[_CTCPathKey, float] = {}

            for key, path_score in self._paths.items():
                blank_score = path_score + float(row[chunk.blank_id])
                blank_key = _CTCPathKey(key.node, key.last_token, True, key.start_frame, key.anchor_token)
                self._merge(next_paths, blank_key, blank_score)

                for token_id in top_ids:
                    if token_id == chunk.blank_id:
                        continue
                    score = path_score + float(row[token_id])
                    if token_id == key.last_token and not key.ended_blank:
                        repeated_key = _CTCPathKey(key.node, key.last_token, False, key.start_frame, key.anchor_token)
                        self._merge(next_paths, repeated_key, score)
                        continue
                    symbol = chunk.symbols[token_id]
                    child = self.trie.child(key.node, symbol)
                    if child is None:
                        continue
                    advanced = _CTCPathKey(child, token_id, False, key.start_frame, key.anchor_token)
                    self._merge(next_paths, advanced, score)
                    self._record_completions(completed, advanced, next_paths[advanced], absolute_frame)

            for token_id in top_ids:
                if token_id == chunk.blank_id:
                    continue
                symbol = chunk.symbols[token_id]
                child = self.trie.child(0, symbol)
                if child is None:
                    continue
                key = _CTCPathKey(child, token_id, False, absolute_frame, anchor_token)
                self._merge(next_paths, key, float(row[token_id]))
                self._record_completions(completed, key, next_paths[key], absolute_frame)

            self._paths = self._prune(next_paths)
        if chunk.log_probs:
            self._last_frame = chunk.frame_offset + len(chunk.log_probs) - 1

        active_by_id: dict[str, CandidateEvidence] = {}
        final_frame = chunk.frame_offset + max(0, len(chunk.log_probs) - 1)
        for key, score in self._paths.items():
            frames = max(1, final_frame - key.start_frame + 1)
            confidence = max(0.0, min(0.999, math.exp(score / frames)))
            for hotword_id in self.trie.candidate_ids(key.node, limit=self.top_k_candidates):
                if not self._enabled(hotword_id) or hotword_id in completed:
                    continue
                evidence = CandidateEvidence(
                    hotword_id=hotword_id,
                    confidence=confidence,
                    start_frame=key.start_frame,
                    anchor_token=key.anchor_token,
                )
                previous = active_by_id.get(hotword_id)
                if previous is None or evidence.confidence > previous.confidence:
                    active_by_id[hotword_id] = evidence

        active = sorted(active_by_id.values(), key=lambda item: item.confidence, reverse=True)[: self.top_k_candidates]
        complete = sorted(completed.values(), key=lambda item: item.confidence, reverse=True)[: self.top_k_candidates]
        return RetrievalUpdate(active=tuple(active), completed=tuple(complete))

    def _record_completions(
        self,
        completed: dict[str, CandidateEvidence],
        key: _CTCPathKey,
        score: float,
        frame: int,
    ) -> None:
        frames = max(1, frame - key.start_frame + 1)
        confidence = max(0.0, min(1.0, math.exp(score / frames)))
        if confidence < self.min_completion_confidence:
            return
        for hotword_id in self.trie.nodes[key.node].terminal_ids:
            if not self._enabled(hotword_id):
                continue
            evidence = CandidateEvidence(
                hotword_id=hotword_id,
                confidence=confidence,
                start_frame=key.start_frame,
                anchor_token=key.anchor_token,
                complete=True,
            )
            previous = completed.get(hotword_id)
            if previous is None or evidence.confidence > previous.confidence:
                completed[hotword_id] = evidence

    def _prune(self, paths: dict[_CTCPathKey, float]) -> dict[_CTCPathKey, float]:
        if not paths:
            return {}
        best = max(paths.values())
        survivors = [(key, score) for key, score in paths.items() if score >= best - self.beam_threshold]
        survivors.sort(key=lambda item: item[1], reverse=True)
        return dict(survivors[: self.max_active_states])

    def reset(self) -> None:
        """Clear all cross-chunk token-passing paths."""

        self._paths.clear()

        self._last_chunk_id = -1
        self._last_frame = -1
