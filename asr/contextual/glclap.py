"""GLCLAP hotword indexing and accumulated-audio retrieval primitives.

The module deliberately depends only on NumPy.  Qwen/PyTorch model loading is
implemented in :mod:`asr.contextual.glclap_model`, which keeps catalog search
and streaming state testable on CPU-only machines.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .replay import ReplayClock

import numpy as np


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked hotword returned by the GLCLAP index."""

    hotword_id: str
    score: float
    rank: int
    peak_frame: int
    variant: str


@dataclass(frozen=True)
class RetrievalBatch:
    """Retrieval output for one accumulated-audio refresh."""

    accumulated_audio_sec: float
    frame_count: int
    hits: tuple[RetrievalHit, ...]
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    chunk_id: int = -1
    is_final: bool = False
    timeline: Mapping[str, float | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for traces and manifests."""

        return asdict(self)


@dataclass(frozen=True)
class AudioEncoding:
    """Normalized audio-frame embeddings and optional stage timings.

    ``frames`` has shape ``[T, D]``.  The runtime normally reports separate
    ``aut_ms``, ``projector_ms``, and ``adapter_ms`` measurements.
    """

    frames: Any
    timings_ms: Mapping[str, float] = field(default_factory=dict)


class AudioFrameEncoder(Protocol):
    """Minimum runtime contract used by accumulated-audio sessions."""

    def encode_pcm(self, pcm16k: Sequence[float]) -> AudioEncoding | Any:
        """Encode all 16 kHz PCM observed so far into ``[T, D]`` frames."""


def _as_float_matrix(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], found {matrix.shape}")
    return matrix


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float32).eps)


class HotwordEmbeddingIndex:
    """Immutable exact-search index over canonical and alias embeddings.

    Search is blockwise over keys, so peak memory is ``O(T * block_size)``
    while the result is identical to a dense matrix multiplication.  Variants
    sharing a hotword ID are max-aggregated before the final Top-K ranking.
    """

    FORMAT_VERSION = "glclap-index-v1"

    def __init__(
        self,
        keys: Any,
        hotword_ids: Sequence[str],
        variants: Sequence[str],
        *,
        block_size: int = 16384,
        catalog_version: str = "1",
        normalize: bool = True,
    ) -> None:
        matrix = _as_float_matrix(keys, "keys")
        if matrix.shape[0] != len(hotword_ids) or matrix.shape[0] != len(variants):
            raise ValueError("keys, hotword_ids, and variants must have equal length")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if normalize:
            matrix = _l2_normalize(matrix)
        # The on-disk/index contract is fp16; scoring promotes blocks to fp32.
        self._keys = np.ascontiguousarray(matrix.astype(np.float16))
        self._torch_keys: Any | None = None
        self._hotword_ids = tuple(str(item) for item in hotword_ids)
        self._variants = tuple(str(item) for item in variants)
        self.block_size = int(block_size)
        self.catalog_version = str(catalog_version)

    @property
    def embedding_dim(self) -> int:
        """Return the key dimension."""

        return int(self._keys.shape[1])

    @property
    def variant_count(self) -> int:
        """Return the number of canonical/alias keys."""

        return int(self._keys.shape[0])

    @property
    def hotword_count(self) -> int:
        """Return the number of unique catalog IDs."""

        return len(set(self._hotword_ids))

    def to(self, device: Any) -> "HotwordEmbeddingIndex":
        """Stage a read-only fp32 key cache on a PyTorch device.

        The serialized index remains fp16. Device staging is optional and is
        used by the CUDA streaming CLI to keep score-only latency bounded.
        """

        try:
            import torch
        except ImportError as exc:
            raise ImportError("device index staging requires PyTorch") from exc
        self._torch_keys = torch.from_numpy(self._keys).to(device=device, dtype=torch.float32)
        return self

    def search(self, audio_frames: Any, top_k: int = 50) -> RetrievalBatch:
        """Run exact max-over-time retrieval and aggregate aliases by ID."""

        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        started = time.perf_counter()
        is_torch = hasattr(audio_frames, "device") and audio_frames.__class__.__module__.startswith("torch")
        if is_torch:
            import torch

            frames_tensor = audio_frames.detach().float()
            if frames_tensor.ndim == 3 and frames_tensor.shape[0] == 1:
                frames_tensor = frames_tensor[0]
            if frames_tensor.ndim != 2:
                raise ValueError(f"audio_frames must have shape [T, D], found {tuple(frames_tensor.shape)}")
            if frames_tensor.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"audio/key dimension mismatch: {frames_tensor.shape[1]} != {self.embedding_dim}"
                )
            frame_count = int(frames_tensor.shape[0])
            if not frame_count or not self.variant_count or top_k == 0:
                elapsed = (time.perf_counter() - started) * 1000.0
                return RetrievalBatch(0.0, frame_count, (), {"search_ms": elapsed})
            frames_tensor = torch.nn.functional.normalize(frames_tensor, p=2, dim=-1)
            if self._torch_keys is None or self._torch_keys.device != frames_tensor.device:
                self.to(frames_tensor.device)
            scores_tensor = torch.empty(self.variant_count, dtype=torch.float32, device=frames_tensor.device)
            peak_tensor = torch.empty(self.variant_count, dtype=torch.long, device=frames_tensor.device)
            for begin in range(0, self.variant_count, self.block_size):
                end = min(self.variant_count, begin + self.block_size)
                similarities = frames_tensor @ self._torch_keys[begin:end].T
                local_scores, local_peak = similarities.max(dim=0)
                scores_tensor[begin:end] = local_scores
                peak_tensor[begin:end] = local_peak
            if frames_tensor.is_cuda:
                torch.cuda.synchronize(frames_tensor.device)
            scores = scores_tensor.cpu().numpy()
            peak_frames = peak_tensor.cpu().numpy()
        else:
            frames = _as_float_matrix(audio_frames, "audio_frames")
            if frames.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"audio/key dimension mismatch: {frames.shape[1]} != {self.embedding_dim}"
                )
            frames = _l2_normalize(frames)
            frame_count = len(frames)
            if not frame_count or not self.variant_count or top_k == 0:
                elapsed = (time.perf_counter() - started) * 1000.0
                return RetrievalBatch(0.0, frame_count, (), {"search_ms": elapsed})
            scores = np.empty(self.variant_count, dtype=np.float32)
            peak_frames = np.empty(self.variant_count, dtype=np.int64)
            for begin in range(0, self.variant_count, self.block_size):
                end = min(self.variant_count, begin + self.block_size)
                similarities = frames @ self._keys[begin:end].astype(np.float32).T
                local_peak = np.argmax(similarities, axis=0)
                columns = np.arange(end - begin)
                scores[begin:end] = similarities[local_peak, columns]
                peak_frames[begin:end] = local_peak

        best_by_id: dict[str, tuple[float, int, str]] = {}
        for index, hotword_id in enumerate(self._hotword_ids):
            candidate = (float(scores[index]), int(peak_frames[index]), self._variants[index])
            previous = best_by_id.get(hotword_id)
            if previous is None or candidate[0] > previous[0] or (
                candidate[0] == previous[0] and candidate[2] < previous[2]
            ):
                best_by_id[hotword_id] = candidate

        ordered = sorted(
            best_by_id.items(),
            key=lambda item: (-item[1][0], item[0], item[1][2]),
        )[:top_k]
        hits = tuple(
            RetrievalHit(
                hotword_id=hotword_id,
                score=payload[0],
                rank=rank,
                peak_frame=payload[1],
                variant=payload[2],
            )
            for rank, (hotword_id, payload) in enumerate(ordered, 1)
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        return RetrievalBatch(0.0, frame_count, hits, {"search_ms": elapsed})

    def save(self, path: str | Path) -> None:
        """Write the fp16 key matrix and JSON metadata to one NPZ file."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": self.FORMAT_VERSION,
            "catalog_version": self.catalog_version,
            "block_size": self.block_size,
            "embedding_dim": self.embedding_dim,
        }
        with target.open("wb") as stream:
            np.savez_compressed(
                stream,
                keys=self._keys,
                hotword_ids=np.asarray(self._hotword_ids, dtype=np.str_),
                variants=np.asarray(self._variants, dtype=np.str_),
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )

    @classmethod
    def load(cls, path: str | Path) -> "HotwordEmbeddingIndex":
        """Load and validate an index written by :meth:`save`."""

        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            if metadata.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError("unsupported GLCLAP index format")
            return cls(
                payload["keys"],
                payload["hotword_ids"].tolist(),
                payload["variants"].tolist(),
                block_size=int(metadata.get("block_size", 16384)),
                catalog_version=str(metadata.get("catalog_version", "1")),
                normalize=False,
            )


class AccumulatedAudioRetrievalSession:
    """Re-encode all accumulated PCM at each fixed-size refresh boundary."""

    def __init__(
        self,
        encoder: AudioFrameEncoder,
        index: HotwordEmbeddingIndex,
        *,
        chunk_size_sec: float = 2.0,
        sample_rate: int = 16000,
        top_k: int = 50,
        refresh_clock: ReplayClock | None = None,
    ) -> None:
        if chunk_size_sec <= 0:
            raise ValueError("chunk_size_sec must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.encoder = encoder
        self._index = index
        self.chunk_size_sec = float(chunk_size_sec)
        self.sample_rate = int(sample_rate)
        self.top_k = int(top_k)
        self.refresh_clock = refresh_clock
        self.chunk_samples = max(1, round(self.chunk_size_sec * self.sample_rate))
        self._audio_accum: list[float] = []
        self._processed_samples = 0
        self._chunk_id = 0
        self._last_batch: RetrievalBatch | None = None
        self._finished = False

    @property
    def index(self) -> HotwordEmbeddingIndex:
        """Return the session index."""

        return self._index

    def set_index(self, index: HotwordEmbeddingIndex) -> None:
        """Replace the index only while the session is in reset state."""

        if self._audio_accum or self._finished:
            raise RuntimeError("reset the session before replacing its index")
        self._index = index

    def step(self, pcm16k: Sequence[float]) -> tuple[RetrievalBatch, ...]:
        """Append PCM and return every newly completed 2-second refresh."""

        if self._finished:
            raise RuntimeError("cannot step a finished session; call reset()")
        self._audio_accum.extend(float(sample) for sample in pcm16k)
        batches: list[RetrievalBatch] = []
        next_boundary = self._processed_samples + self.chunk_samples
        while len(self._audio_accum) >= next_boundary:
            batches.append(self._refresh(next_boundary, is_final=False))
            self._processed_samples = next_boundary
            next_boundary += self.chunk_samples
        return tuple(batches)

    def finish(self) -> RetrievalBatch:
        """Process the tail block and mark the returned result final."""

        if self._finished:
            if self._last_batch is None:
                raise RuntimeError("finished session has no retrieval result")
            return self._last_batch
        if len(self._audio_accum) > self._processed_samples or self._last_batch is None:
            batch = self._refresh(len(self._audio_accum), is_final=True)
            self._processed_samples = len(self._audio_accum)
        else:
            batch = replace(self._last_batch, is_final=True)
            self._last_batch = batch
        self._finished = True
        return batch

    def _refresh(self, sample_count: int, *, is_final: bool) -> RetrievalBatch:
        stamp = self.refresh_clock.start(sample_count) if self.refresh_clock else None
        encode_started = time.perf_counter()
        encoded = self.encoder.encode_pcm(self._audio_accum[:sample_count])
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        if isinstance(encoded, AudioEncoding):
            frames = encoded.frames
            timings = dict(encoded.timings_ms)
        else:
            frames = encoded
            timings = {}
        timings["encode_ms"] = encode_ms
        found = self._index.search(frames, top_k=self.top_k)
        timeline = self.refresh_clock.finish(stamp) if stamp is not None else {}
        timings.update(found.timings_ms)
        timings["total_ms"] = encode_ms + float(found.timings_ms.get("search_ms", 0.0))
        timings["processing_ms"] = (float(timeline["processing_sec"]) * 1000 if timeline
                                     else (time.perf_counter() - encode_started) * 1000)
        batch = RetrievalBatch(
            accumulated_audio_sec=sample_count / self.sample_rate,
            frame_count=found.frame_count,
            hits=found.hits,
            timings_ms=timings,
            chunk_id=self._chunk_id,
            is_final=is_final,
            timeline=timeline,
        )
        self._chunk_id += 1
        self._last_batch = batch
        return batch

    def reset(self) -> None:
        """Clear all stream-local audio and timing state."""

        if self.refresh_clock is not None:
            self.refresh_clock.reset()
        self._audio_accum = []
        self._processed_samples = 0
        self._chunk_id = 0
        self._last_batch = None
        self._finished = False
