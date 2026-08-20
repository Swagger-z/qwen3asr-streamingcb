"""Text and optional PyTorch acoustic probes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable, Protocol, Sequence

from .catalog import HotwordCatalog, TokenizerLike, normalize_text, text_variants
from .retriever import AhoCorasickRetriever
from .types import CandidateEvidence, ProbeChunk, RetrievalUpdate


class AcousticProbe(Protocol):
    """Interface for a sidecar acoustic posterior producer."""

    def process_pcm(self, pcm16k: Sequence[float], chunk_id: int) -> ProbeChunk | None:
        """Process one new waveform chunk."""

    def reset(self) -> None:
        """Reset all stream-local state."""


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


class TranscriptProbe:
    """No-training matcher over provisional transcript revisions.

    The probe feeds only the newly appended normalized suffix when the previous
    hypothesis remains a prefix. If an earlier character is revised, it resets
    and replays the current hypothesis, preventing stale cross-chunk paths.
    """

    def __init__(
        self,
        catalog: HotwordCatalog,
        enabled_ids: Iterable[str] | None = None,
        stateful: bool = True,
        top_k_candidates: int = 20,
    ) -> None:
        self.catalog = catalog
        self.enabled_ids = set(enabled_ids) if enabled_ids is not None else set(catalog.entries)
        self.stateful = stateful
        self.retriever = AhoCorasickRetriever(
            catalog.text_trie,
            self.enabled_ids,
            stateful=stateful,
            top_k_candidates=top_k_candidates,
        )
        self._last_text = ""

    def observe(
        self,
        text: str,
        tokenizer: TokenizerLike,
        chunk_id: int,
        default_anchor_token: int,
    ) -> RetrievalUpdate:
        """Observe a refreshed full transcript and emit contextual evidence."""

        normalized = normalize_text(text)
        common = _common_prefix_length(self._last_text, normalized)
        if not self.stateful:
            delta = normalized[common:]
            frame_offset = common
        elif common < len(self._last_text):
            self.retriever.reset()
            delta = normalized
            frame_offset = 0
        else:
            delta = normalized[common:]
            frame_offset = common
        update = self.retriever.consume(
            tuple(delta),
            frame_offset=frame_offset,
            anchor_token=default_anchor_token,
            chunk_id=chunk_id,
        )
        self._last_text = normalized
        return self._refine_anchors(update, normalized, tokenizer)

    def _refine_anchors(
        self,
        update: RetrievalUpdate,
        normalized_text: str,
        tokenizer: TokenizerLike,
    ) -> RetrievalUpdate:
        def refine(evidence: CandidateEvidence) -> CandidateEvidence:
            if evidence.complete:
                starts = []
                for variant in text_variants(self.catalog.entries[evidence.hotword_id]):
                    index = normalized_text.rfind(normalize_text(variant))
                    if index >= 0:
                        starts.append(index)
                char_start = max(starts) if starts else max(0, evidence.start_frame or 0)
            else:
                depth = self.retriever.trie.nodes[self.retriever.node].depth
                char_start = max(0, len(normalized_text) - depth)
            anchor = len(tuple(tokenizer.encode(normalized_text[:char_start])))
            return replace(evidence, anchor_token=anchor)

        return RetrievalUpdate(
            active=tuple(refine(item) for item in update.active),
            completed=tuple(refine(item) for item in update.completed),
            rejected_ids=update.rejected_ids,
        )

    def update_enabled(self, enabled_ids: Iterable[str]) -> None:
        """Replace allowed IDs without replaying old transcript."""

        self.enabled_ids = set(enabled_ids)
        self.retriever.update_enabled(self.enabled_ids)

    def replace_catalog(self, catalog: HotwordCatalog, enabled_ids: Iterable[str]) -> None:
        """Replace a catalog after an overlay update and reset matcher state."""

        self.catalog = catalog
        self.enabled_ids = set(enabled_ids)
        self.retriever = AhoCorasickRetriever(
            catalog.text_trie,
            self.enabled_ids,
            stateful=self.stateful,
            top_k_candidates=self.retriever.top_k_candidates,
        )
        self._last_text = ""

    def reset(self) -> None:
        """Clear transcript and automaton state."""

        self.retriever.reset()
        self._last_text = ""


try:  # Optional GPU dependency.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - dependency-free environments.
    torch = None
    nn = None


if nn is not None:

    class PhonemeCTCHead(nn.Module):
        """LayerNorm plus linear phoneme projection over AuT states."""

        def __init__(self, encoder_dim: int, vocab_size: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(encoder_dim)
            self.projection = nn.Linear(encoder_dim, vocab_size)

        def forward(self, hidden_states: Any) -> Any:
            """Project ``[B, T, D]`` states to ``[B, T, P]`` logits."""

            return self.projection(self.norm(hidden_states))

else:

    class PhonemeCTCHead:  # type: ignore[no-redef]
        """Dependency error shim when PyTorch is unavailable."""

        def __init__(self, encoder_dim: int, vocab_size: int) -> None:
            del encoder_dim, vocab_size
            raise ImportError("PhonemeCTCHead requires PyTorch; install the 'probe' extra")


def resolve_qwen_audio_encoder(model: Any) -> Any:
    """Resolve common Qwen audio-encoder attribute layouts."""

    candidates = (
        getattr(model, "audio_tower", None),
        getattr(model, "audio_encoder", None),
        getattr(getattr(model, "model", None), "audio_tower", None),
        getattr(getattr(model, "model", None), "audio_encoder", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise AttributeError("unable to locate Qwen audio encoder; pinned adapter contract changed")


class AuTPhonemeProbe:
    """Frozen AuT sidecar plus trainable phoneme CTC head.

    ``feature_extractor`` converts 16 kHz PCM into the inputs expected by the
    encoder. It may return a tensor, positional argument tuple, or keyword
    dictionary. The probe keeps 200 ms left PCM overlap by default and removes
    the corresponding posterior frames from its output.
    """

    def __init__(
        self,
        encoder: Any,
        head: Any,
        feature_extractor: Callable[[Sequence[float]], Any],
        symbols: Sequence[str],
        blank_id: int,
        sample_rate: int = 16000,
        frame_rate_hz: float = 12.5,
        left_overlap_ms: int = 200,
    ) -> None:
        if torch is None:
            raise ImportError("AuTPhonemeProbe requires PyTorch; install the 'probe' extra")
        self.encoder = encoder
        self.head = head
        self.feature_extractor = feature_extractor
        self.symbols = tuple(symbols)
        self.blank_id = blank_id
        self.sample_rate = sample_rate
        self.frame_rate_hz = frame_rate_hz
        self.left_overlap_samples = round(sample_rate * left_overlap_ms / 1000)
        self._overlap: list[float] = []
        self._processed_frames = 0
        if hasattr(self.encoder, "parameters"):
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        if hasattr(self.encoder, "eval"):
            self.encoder.eval()

    def process_pcm(self, pcm16k: Sequence[float], chunk_id: int) -> ProbeChunk | None:
        """Return only posterior frames not attributable to left overlap."""

        del chunk_id
        current = [float(sample) for sample in pcm16k]
        combined = self._overlap + current
        inputs = self.feature_extractor(combined)
        with torch.no_grad():
            if isinstance(inputs, dict):
                output = self.encoder(**inputs)
            elif isinstance(inputs, tuple):
                output = self.encoder(*inputs)
            else:
                output = self.encoder(inputs)
            hidden = self._extract_hidden(output)
            logits = self.head(hidden)
            log_probs = torch.log_softmax(logits, dim=-1)
        if log_probs.ndim == 3:
            log_probs = log_probs[0]
        overlap_frames = min(log_probs.shape[0], round(len(self._overlap) / self.sample_rate * self.frame_rate_hz))
        new_probs = log_probs[overlap_frames:].detach().float().cpu().tolist()
        frame_offset = self._processed_frames
        self._processed_frames += len(new_probs)
        self._overlap = combined[-self.left_overlap_samples :] if self.left_overlap_samples else []
        if not new_probs:
            return None
        return ProbeChunk(new_probs, frame_offset, self.blank_id, self.symbols)

    @staticmethod
    def _extract_hidden(output: Any) -> Any:
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, dict):
            for key in ("last_hidden_state", "hidden_states", "audio_features"):
                if key in output:
                    value = output[key]
                    return value[-1] if key == "hidden_states" and isinstance(value, (tuple, list)) else value
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        return output

    def reset(self) -> None:
        """Clear overlap and absolute frame counters."""

        self._overlap = []
        self._processed_frames = 0
