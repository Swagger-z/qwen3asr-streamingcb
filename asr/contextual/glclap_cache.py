"""Caches for deterministic frozen Qwen features used by GLCLAP training."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

from .glclap_model import QwenAudioFeatures

try:
    import torch
except ImportError:  # pragma: no cover - optional training dependency.
    torch = None


class QwenFeatureCache:
    """Disk-backed cache for frozen pre- or post-projector audio features."""

    FORMAT_VERSION = "qwen-glclap-feature-v1"

    def __init__(
        self,
        root: str | Path,
        *,
        namespace: str,
        feature_kind: str,
        device: Any,
        dtype: Any,
    ) -> None:
        if feature_kind not in {"pre_projector", "post_projector"}:
            raise ValueError("feature_kind must be pre_projector or post_projector")
        self.root = Path(root)
        self.namespace = str(namespace)
        self.feature_kind = feature_kind
        self.device = device
        self.dtype = dtype
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _path(self, key: str, source: str | Path) -> Path:
        source_path = Path(source)
        stat = source_path.stat()
        identity = (
            f"{self.namespace}|{key}|{source_path.resolve()}|"
            f"{stat.st_size}|{stat.st_mtime_ns}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.root / self.feature_kind / digest[:2] / f"{digest}.pt"

    def get(self, key: str, source: str | Path) -> QwenAudioFeatures | None:
        """Load one cached feature sequence, returning None on a miss."""

        if torch is None:
            raise ImportError("QwenFeatureCache requires PyTorch")
        path = self._path(key, source)
        if not path.is_file():
            self.misses += 1
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("format_version") != self.FORMAT_VERSION
            or payload.get("namespace") != self.namespace
            or payload.get("feature_kind") != self.feature_kind
        ):
            self.misses += 1
            return None
        value = payload["frames"].to(device=self.device, dtype=self.dtype, non_blocking=True)
        empty = torch.empty((0, 0), device=self.device, dtype=self.dtype)
        self.hits += 1
        if self.feature_kind == "post_projector":
            return QwenAudioFeatures(empty, value)
        return QwenAudioFeatures(value, empty)

    def put(self, key: str, source: str | Path, features: QwenAudioFeatures) -> None:
        """Atomically store the feature branch required by the training mode."""

        value = (
            features.post_projector
            if self.feature_kind == "post_projector"
            else features.pre_projector
        )
        self.put_tensor(key, source, value)

    def put_tensor(self, key: str, source: str | Path, value: Any) -> None:
        """Store an already selected feature tensor, preferably on CPU."""

        if torch is None:
            raise ImportError("QwenFeatureCache requires PyTorch")
        path = self._path(key, source)
        if path.is_file():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        payload = {
            "format_version": self.FORMAT_VERSION,
            "namespace": self.namespace,
            "feature_kind": self.feature_kind,
            "frames": value.detach().to(device="cpu"),
        }
        torch.save(payload, temporary)
        try:
            os.replace(temporary, path)
            self.writes += 1
        finally:
            temporary.unlink(missing_ok=True)

    def stats(self) -> dict[str, int]:
        """Return cumulative hit/miss/write counters."""

        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}


class FrozenTextEmbeddingCache:
    """Bounded device cache for mean-pooled frozen Qwen token embeddings."""

    def __init__(self, max_entries: int = 20000, encode_batch_size: int = 1024) -> None:
        if max_entries < 0 or encode_batch_size <= 0:
            raise ValueError("invalid text cache limits")
        self.max_entries = int(max_entries)
        self.encode_batch_size = int(encode_batch_size)
        self._values: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _encode_missing(
        self,
        model: Any,
        processor: Any,
        texts: Sequence[str],
    ) -> None:
        if torch is None:
            raise ImportError("FrozenTextEmbeddingCache requires PyTorch")
        device = next(model.adapters.parameters()).device
        unique = list(dict.fromkeys(texts))
        for begin in range(0, len(unique), self.encode_batch_size):
            batch_texts = unique[begin : begin + self.encode_batch_size]
            tokens = processor.tokenizer(
                batch_texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                pooled = model.encoder.embed_text(
                    tokens["input_ids"].to(device),
                    tokens["attention_mask"].to(device),
                )
            for text, value in zip(batch_texts, pooled):
                self._values[text] = value.detach()
                self._values.move_to_end(text)
                while self.max_entries and len(self._values) > self.max_entries:
                    self._values.popitem(last=False)

    def get(self, model: Any, processor: Any, texts: Sequence[str]) -> Any:
        """Return cached pooled embeddings in the requested text order."""

        if torch is None:
            raise ImportError("FrozenTextEmbeddingCache requires PyTorch")
        if not texts:
            dimension = int(model.encoder.post_projector_dim)
            return torch.empty(
                (0, dimension),
                device=next(model.adapters.parameters()).device,
                dtype=next(model.encoder.token_embedding.parameters()).dtype,
            )
        if self.max_entries == 0:
            tokens = processor.tokenizer(
                list(texts),
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            device = next(model.adapters.parameters()).device
            return model.encoder.embed_text(
                tokens["input_ids"].to(device),
                tokens["attention_mask"].to(device),
            )
        missing = [text for text in dict.fromkeys(texts) if text not in self._values]
        self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        if missing:
            self._encode_missing(model, processor, missing)
        output = []
        for text in texts:
            value = self._values[text]
            self._values.move_to_end(text)
            output.append(value)
        return torch.stack(output, dim=0)

    def stats(self) -> dict[str, int]:
        """Return cumulative cache counters and current size."""

        return {"hits": self.hits, "misses": self.misses, "entries": len(self._values)}
