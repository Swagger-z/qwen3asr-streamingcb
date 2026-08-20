"""End-to-end contextual streaming session orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from asr.backends.qwen3 import StreamingLLMBackend

from .candidates import CandidateManager
from .catalog import HotwordCatalog
from .commit import CommitController
from .probes import AcousticProbe, TranscriptProbe
from .retriever import CTCPosteriorTrieRetriever
from .token_bias import SparseTokenBias
from .trace import TraceWriter
from .types import HotwordEntry, RetrievalUpdate, StreamingResult


@dataclass(frozen=True)
class ContextualSessionConfig:
    """Configurable streaming, retrieval, and commitment policy."""

    sample_rate: int = 16000
    chunk_size_ms: int = 1000
    base_rollback_tokens: int = 5
    max_rollback_tokens: int = 32
    unfixed_chunk_num: int = 2
    enter_threshold: float = 0.55
    exit_threshold: float = 0.35
    active_ttl_chunks: int = 3
    confirmed_ttl_chunks: int = 2
    retrieval_top_k: int = 20
    prompt_top_k: int = 5
    token_bias_bonus: float = 2.0
    token_bias_max_bonus: float = 4.0
    enable_token_bias: bool = True
    enable_transcript_probe: bool = True
    stateful_transcript_probe: bool = True
    enable_adaptive_lookahead: bool = False
    lookahead_ms: int = 200
    uncertainty_low: float = 0.45
    uncertainty_high: float = 0.75
    max_effective_context_ms: int = 2000

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.chunk_size_ms <= 0:
            raise ValueError("sample_rate and chunk_size_ms must be positive")
        if self.lookahead_ms not in (100, 200, 300):
            raise ValueError("lookahead_ms must be one of 100, 200, or 300")
        if self.enable_adaptive_lookahead and self.chunk_size_ms + self.lookahead_ms > self.max_effective_context_ms:
            raise ValueError("chunk plus lookahead exceeds max effective context")


class ContextualStreamingSession:
    """Coordinate accumulated-audio Qwen refreshes and contextual state."""

    def __init__(
        self,
        backend: StreamingLLMBackend,
        catalog: HotwordCatalog,
        config: ContextualSessionConfig | None = None,
        enabled_ids: Iterable[str] | None = None,
        static_context: str = "",
        acoustic_probe: AcousticProbe | None = None,
        acoustic_retriever: CTCPosteriorTrieRetriever | None = None,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self.backend = backend
        self.catalog = catalog
        self.config = config or ContextualSessionConfig()
        self.enabled_ids = set(enabled_ids) if enabled_ids is not None else set(catalog.entries)
        self.static_context = static_context
        self.acoustic_probe = acoustic_probe
        self.acoustic_retriever = acoustic_retriever
        if (acoustic_probe is None) != (acoustic_retriever is None):
            raise ValueError("acoustic_probe and acoustic_retriever must be provided together")
        self.trace_writer = trace_writer
        self.candidates = self._new_candidate_manager()
        self.commit = CommitController(
            self.config.base_rollback_tokens,
            self.config.max_rollback_tokens,
            self.config.unfixed_chunk_num,
        )
        self.transcript_probe = (
            TranscriptProbe(
                catalog,
                self.enabled_ids,
                stateful=self.config.stateful_transcript_probe,
                top_k_candidates=self.config.retrieval_top_k,
            )
            if self.config.enable_transcript_probe
            else None
        )
        self.token_bias = SparseTokenBias(
            catalog,
            backend.tokenizer,
            self.enabled_ids,
            self.config.token_bias_bonus,
            self.config.token_bias_max_bonus,
        )
        self._buffer: list[float] = []
        self._audio_accum: list[float] = []
        self._raw_text = ""
        self._display_text = ""
        self._language = ""
        self._raw_token_ids: list[int] = []
        self._chunk_id = 0
        self._revision_count = 0
        self._last_result = self._empty_result()
        self._pending_enabled_ids: set[str] | None = None
        self._pending_overlay: tuple[HotwordEntry, ...] = ()

    def _new_candidate_manager(self) -> CandidateManager:
        return CandidateManager(
            self.catalog,
            self.enabled_ids,
            self.config.enter_threshold,
            self.config.exit_threshold,
            self.config.active_ttl_chunks,
            self.config.confirmed_ttl_chunks,
            self.config.retrieval_top_k,
            self.config.prompt_top_k,
        )

    @property
    def chunk_id(self) -> int:
        """Return the next zero-based chunk identifier."""

        return self._chunk_id

    def step(self, pcm16k: Sequence[float]) -> StreamingResult:
        """Buffer arbitrary PCM and process every complete effective chunk."""

        self._buffer.extend(float(sample) for sample in pcm16k)
        results: list[StreamingResult] = []
        committed_parts: list[str] = []
        while len(self._buffer) >= self._required_samples():
            required = self._required_samples()
            chunk = self._buffer[:required]
            del self._buffer[:required]
            result = self._process_chunk(chunk)
            results.append(result)
            committed_parts.append(result.committed_delta)
        if not results:
            debug = dict(self._last_result.debug)
            debug["buffered_samples"] = len(self._buffer)
            return StreamingResult(
                partial_text=self._last_result.partial_text,
                stable_text=self._last_result.stable_text,
                committed_delta="",
                active_candidates=self._last_result.active_candidates,
                confirmed_candidates=self._last_result.confirmed_candidates,
                rollback_start_token=self._last_result.rollback_start_token,
                chunk_id=self._last_result.chunk_id,
                processing_ms=0.0,
                is_final=False,
                debug=debug,
            )
        latest = results[-1]
        if len(results) == 1:
            return latest
        return StreamingResult(
            partial_text=latest.partial_text,
            stable_text=latest.stable_text,
            committed_delta="".join(committed_parts),
            active_candidates=latest.active_candidates,
            confirmed_candidates=latest.confirmed_candidates,
            rollback_start_token=latest.rollback_start_token,
            chunk_id=latest.chunk_id,
            processing_ms=sum(item.processing_ms for item in results),
            is_final=False,
            debug=latest.debug,
        )

    def _required_samples(self) -> int:
        base = round(self.config.sample_rate * self.config.chunk_size_ms / 1000)
        if not self.config.enable_adaptive_lookahead:
            return base
        uncertain = any(
            self.config.uncertainty_low <= item.confidence <= self.config.uncertainty_high
            for item in self.candidates.holding_candidates()
        )
        if not uncertain:
            return base
        return base + round(self.config.sample_rate * self.config.lookahead_ms / 1000)

    def _process_chunk(self, chunk: Sequence[float]) -> StreamingResult:
        started = time.perf_counter()
        self._apply_pending_updates()
        self._audio_accum.extend(chunk)
        previous_ids = list(self._raw_token_ids)
        previous_count = len(previous_ids)

        if self.acoustic_probe is not None and self.acoustic_retriever is not None:
            probe_chunk = self.acoustic_probe.process_pcm(chunk, self._chunk_id)
            if probe_chunk is not None:
                update = self.acoustic_retriever.consume(probe_chunk, previous_count, self._chunk_id)
                self.candidates.apply(update, self._chunk_id, previous_count)
        else:
            self.candidates.apply(RetrievalUpdate(), self._chunk_id, previous_count)

        holding_before = self.candidates.holding_candidates()
        decode_rollback = self.commit.rollback_start(
            previous_count,
            (item.anchor_token for item in holding_before),
            self._chunk_id,
        )
        prefix_text = self.backend.tokenizer.decode(previous_ids[:decode_rollback])
        context = self._render_context()
        confidence = self.candidates.confidence_by_id()
        logits_processor = (
            self.token_bias.processor(confidence)
            if self.config.enable_token_bias and confidence
            else None
        )
        decoded = self.backend.decode_accumulated(
            self._audio_accum,
            context=context,
            prefix_text=prefix_text,
            logits_processor=logits_processor,
        )
        new_ids = [int(token) for token in self.backend.tokenizer.encode(decoded.raw_text)]

        if self.transcript_probe is not None:
            text_update = self.transcript_probe.observe(
                decoded.text,
                self.backend.tokenizer,
                self._chunk_id,
                previous_count,
            )
            self.candidates.apply(text_update, self._chunk_id, previous_count)

        holding_after = self.candidates.holding_candidates()
        next_rollback = self.commit.rollback_start(
            len(new_ids),
            (item.anchor_token for item in holding_after),
            self._chunk_id,
        )
        delta_ids = self.commit.commit_after_decode(
            new_ids,
            next_rollback,
            bool(holding_after),
            self._chunk_id,
        )
        revisions = len(previous_ids) - self._token_lcp(previous_ids, new_ids)
        self._revision_count += max(0, revisions)
        self._raw_token_ids = new_ids
        self._raw_text = decoded.raw_text
        self._display_text = decoded.text
        self._language = decoded.language
        stable_text = self.backend.tokenizer.decode(self.commit.committed_ids)
        committed_delta = self.backend.tokenizer.decode(delta_ids)
        elapsed_ms = (time.perf_counter() - started) * 1000
        result = StreamingResult(
            partial_text=decoded.text,
            stable_text=stable_text,
            committed_delta=committed_delta,
            active_candidates=holding_after,
            confirmed_candidates=self.candidates.confirmed_candidates(),
            rollback_start_token=next_rollback,
            chunk_id=self._chunk_id,
            processing_ms=elapsed_ms,
            debug={
                "audio_samples": len(self._audio_accum),
                "audio_end_ms": len(self._audio_accum) / self.config.sample_rate * 1000,
                "buffered_samples": len(self._buffer),
                "decode_rollback_start_token": decode_rollback,
                "raw_decoded": decoded.raw_text,
                "raw_token_ids": new_ids,
                "language": decoded.language,
                "revision_count_total": self._revision_count,
                "prompt_hotword_ids": [entry.id for entry in self.candidates.prompt_entries()],
                "prompt_token_count": len(tuple(self.backend.tokenizer.encode(context))),
                "active_state_count": (
                    self.acoustic_retriever.active_state_count if self.acoustic_retriever is not None else 0
                ),
            },
        )
        self._last_result = result
        if self.trace_writer is not None:
            self.trace_writer.write(result.to_dict())
        self._chunk_id += 1
        return result

    def _render_context(self) -> str:
        entries = self.candidates.prompt_entries()
        if not entries:
            return self.static_context
        hotwords = "; ".join(entry.text for entry in entries)
        dynamic = f"Hotwords: {hotwords}"
        return f"{self.static_context}\n{dynamic}".strip()

    @staticmethod
    def _token_lcp(left: Sequence[int], right: Sequence[int]) -> int:
        index = 0
        limit = min(len(left), len(right))
        while index < limit and left[index] == right[index]:
            index += 1
        return index

    def finish(self) -> StreamingResult:
        """Process residual PCM and commit the entire final hypothesis."""

        processing_ms = 0.0
        if self._buffer:
            tail = list(self._buffer)
            self._buffer.clear()
            tail_result = self._process_chunk(tail)
            processing_ms += tail_result.processing_ms
        delta_ids = self.commit.finish(self._raw_token_ids)
        stable_text = self.backend.tokenizer.decode(self.commit.committed_ids)
        delta = self.backend.tokenizer.decode(delta_ids)
        final = StreamingResult(
            partial_text=self._display_text,
            stable_text=stable_text,
            committed_delta=delta,
            active_candidates=(),
            confirmed_candidates=self.candidates.confirmed_candidates(),
            rollback_start_token=len(self._raw_token_ids),
            chunk_id=max(0, self._chunk_id - 1),
            processing_ms=processing_ms,
            is_final=True,
            debug={
                "audio_samples": len(self._audio_accum),
                "revision_count_total": self._revision_count,
                "language": self._language,
            },
        )
        self._last_result = final
        if self.trace_writer is not None:
            self.trace_writer.write(final.to_dict())
        return final

    def update_hotwords(
        self,
        enabled_ids: Iterable[str] | None = None,
        overlay_entries: Iterable[HotwordEntry] = (),
    ) -> None:
        """Schedule a catalog/filter update for the next processed chunk."""

        self._pending_enabled_ids = set(enabled_ids) if enabled_ids is not None else set(self.enabled_ids)
        self._pending_overlay = tuple(overlay_entries)

    def _apply_pending_updates(self) -> None:
        if self._pending_enabled_ids is None and not self._pending_overlay:
            return
        if self._pending_overlay:
            self.catalog = self.catalog.with_overlay(self._pending_overlay)
        enabled = self._pending_enabled_ids if self._pending_enabled_ids is not None else set(self.enabled_ids)
        enabled |= {entry.id for entry in self._pending_overlay}
        self.enabled_ids = enabled
        self.candidates.catalog = self.catalog
        self.candidates.update_enabled(enabled, self._chunk_id)
        self.token_bias.update_catalog(self.catalog, enabled)
        if self.transcript_probe is not None:
            self.transcript_probe.replace_catalog(self.catalog, enabled)
        if self.acoustic_retriever is not None:
            self.acoustic_retriever.trie = self.catalog.pronunciation_trie
            self.acoustic_retriever.update_enabled(enabled)
            self.acoustic_retriever.reset()
        self._pending_enabled_ids = None
        self._pending_overlay = ()

    def reset(self) -> None:
        """Reset every audio, decoder, retrieval, and commitment state."""

        self._buffer.clear()
        self._audio_accum.clear()
        self._raw_text = ""
        self._display_text = ""
        self._language = ""
        self._raw_token_ids.clear()
        self._chunk_id = 0
        self._revision_count = 0
        self.candidates.reset()
        self.commit.reset()
        if self.transcript_probe is not None:
            self.transcript_probe.reset()
        if self.acoustic_probe is not None:
            self.acoustic_probe.reset()
        if self.acoustic_retriever is not None:
            self.acoustic_retriever.reset()
        self._last_result = self._empty_result()

    @staticmethod
    def _empty_result() -> StreamingResult:
        return StreamingResult(
            partial_text="",
            stable_text="",
            committed_delta="",
            active_candidates=(),
            confirmed_candidates=(),
            rollback_start_token=0,
            chunk_id=-1,
            processing_ms=0.0,
            is_final=False,
            debug={},
        )
