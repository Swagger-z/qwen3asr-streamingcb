"""Hotword-aware token rollback and stable commitment."""

from __future__ import annotations

from typing import Iterable, Sequence


class CommitInvariantError(RuntimeError):
    """Raised when a backend rewrites an already committed prefix."""


class CommitController:
    """Track committed tokens and calculate bounded adaptive rollback."""

    def __init__(
        self,
        base_rollback_tokens: int = 5,
        max_rollback_tokens: int = 32,
        unfixed_chunk_num: int = 2,
    ) -> None:
        if base_rollback_tokens < 0:
            raise ValueError("base_rollback_tokens must be non-negative")
        if max_rollback_tokens < base_rollback_tokens:
            raise ValueError("max rollback must be >= base rollback")
        self.base_rollback_tokens = base_rollback_tokens
        self.max_rollback_tokens = max_rollback_tokens
        self.unfixed_chunk_num = unfixed_chunk_num
        self._committed_ids: list[int] = []

    @property
    def committed_ids(self) -> tuple[int, ...]:
        """Return the immutable committed prefix."""

        return tuple(self._committed_ids)

    def rollback_start(
        self,
        previous_token_count: int,
        active_anchors: Iterable[int],
        chunk_id: int,
    ) -> int:
        """Calculate the prefix length to preserve for the next refresh.

        With active anchors ``A`` the implemented rule is::

            max(C, N-K_max, min(N-K_base, min(A)))

        where ``C`` is the committed prefix length and ``N`` is the previous
        hypothesis length. Before ``unfixed_chunk_num`` the prefix is empty.
        """

        if previous_token_count < len(self._committed_ids):
            raise CommitInvariantError("previous hypothesis is shorter than committed prefix")
        if chunk_id < self.unfixed_chunk_num:
            return len(self._committed_ids)
        baseline = max(len(self._committed_ids), min(1, previous_token_count), previous_token_count - self.base_rollback_tokens)
        anchors = [max(0, min(previous_token_count, int(anchor))) for anchor in active_anchors]
        if not anchors:
            return baseline
        desired = min(baseline, min(anchors))
        return max(len(self._committed_ids), previous_token_count - self.max_rollback_tokens, desired)

    def commit_after_decode(
        self,
        new_token_ids: Sequence[int],
        rollback_start: int,
        has_active_candidate: bool,
        chunk_id: int,
    ) -> tuple[int, ...]:
        """Advance the stable prefix and return newly committed token IDs."""

        new_ids = [int(token) for token in new_token_ids]
        committed_len = len(self._committed_ids)
        if new_ids[:committed_len] != self._committed_ids:
            raise CommitInvariantError("backend rewrote an already committed prefix")
        if chunk_id < self.unfixed_chunk_num:
            target = committed_len
        elif has_active_candidate:
            target = max(committed_len, min(len(new_ids), rollback_start))
        else:
            target = max(committed_len, len(new_ids) - self.base_rollback_tokens)
        delta = tuple(new_ids[committed_len:target])
        self._committed_ids = new_ids[:target]
        return delta

    def finish(self, final_token_ids: Sequence[int]) -> tuple[int, ...]:
        """Commit the complete final hypothesis."""

        final_ids = [int(token) for token in final_token_ids]
        committed_len = len(self._committed_ids)
        if final_ids[:committed_len] != self._committed_ids:
            raise CommitInvariantError("final hypothesis rewrote an already committed prefix")
        delta = tuple(final_ids[committed_len:])
        self._committed_ids = final_ids
        return delta

    def reset(self) -> None:
        """Clear the stable prefix for a new utterance."""

        self._committed_ids.clear()
