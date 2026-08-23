"""PyTorch/Qwen model components for GLCLAP retrieval.

Imports are optional so the rest of :mod:`asr.contextual` remains usable in a
dependency-light environment.  The official Qwen module is treated as a
read-only feature source; warm-start and random projectors are independent
retrieval-branch parameters.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata as metadata
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .glclap import AudioEncoding

try:  # Optional Linux CUDA dependency.
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - exercised by dependency-light tests.
    torch = None
    nn = None
    F = None


EXPECTED_QWEN_ASR_VERSION = "0.0.6"
EXPECTED_TRANSFORMERS_VERSION = "4.57.6"


def _children_for_resolution(value: Any) -> tuple[Any, ...]:
    return tuple(
        child
        for child in (
            getattr(value, "model", None),
            getattr(value, "thinker", None),
            getattr(getattr(value, "model", None), "thinker", None),
        )
        if child is not None and child is not value
    )


def resolve_qwen_thinker(model: Any) -> Any:
    """Resolve the official conditional-generation thinker across wrappers."""

    queue = [model]
    visited: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if getattr(candidate, "audio_tower", None) is not None and (
            hasattr(candidate, "get_input_embeddings")
            or getattr(getattr(candidate, "model", None), "embed_tokens", None) is not None
        ):
            return candidate
        queue.extend(_children_for_resolution(candidate))
    raise AttributeError("unable to locate Qwen thinker/audio_tower; pinned contract changed")


def resolve_qwen_token_embedding(thinker: Any) -> Any:
    """Resolve the frozen LLM input embedding from a Qwen thinker."""

    if hasattr(thinker, "get_input_embeddings"):
        embedding = thinker.get_input_embeddings()
        if embedding is not None:
            return embedding
    embedding = getattr(getattr(thinker, "model", None), "embed_tokens", None)
    if embedding is not None:
        return embedding
    raise AttributeError("unable to locate Qwen LLM token embedding")


def require_qwen_versions() -> None:
    """Validate the pinned Transformers feature-extraction runtime."""

    for package, expected in (
        ("qwen-asr", EXPECTED_QWEN_ASR_VERSION),
        ("transformers", EXPECTED_TRANSFORMERS_VERSION),
    ):
        actual = metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package}=={expected} required, found {actual}")


def load_qwen_transformers(
    model_name_or_path: str = "Qwen/Qwen3-ASR-0.6B",
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load the pinned official Transformers model and processor."""

    if torch is None:
        raise ImportError("Qwen GLCLAP requires PyTorch; install the 'probe' and 'qwen' extras")
    require_qwen_versions()
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:  # pragma: no cover - optional runtime.
        raise ImportError("install qwen-asr==0.0.6") from exc
    wrapper = Qwen3ASRModel.from_pretrained(model_name_or_path, **kwargs)
    return wrapper.model, wrapper.processor


def module_parameter_hash(module: Any) -> str:
    """Hash parameter names, dtypes, shapes, and bytes for mutation tests."""

    if torch is None:
        raise ImportError("module_parameter_hash requires PyTorch")
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class QwenAudioFeatures:
    """Read-only pre/post-projector Qwen frame features."""

    pre_projector: Any
    post_projector: Any
    timings_ms: Mapping[str, float] = field(default_factory=dict)


if nn is not None:

    def _sync(value: Any) -> None:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            torch.cuda.synchronize(value.device)


    def _last_hidden(output: Any) -> Any:
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, Mapping):
            for key in ("last_hidden_state", "audio_features", "hidden_states"):
                if key in output:
                    value = output[key]
                    return value[-1] if isinstance(value, (tuple, list)) else value
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        return output


    class QwenGLCLAPEncoder(nn.Module):
        """Frozen Qwen AuT/projector and LLM token-embedding feature source.

        The audio tower's official forward is left untouched.  A temporary
        forward hook on ``ln_post`` exposes ``[T, 896]`` states, while the
        official return value exposes post-projector ``[T, 1024]`` states.
        """

        def __init__(self, qwen_model: Any) -> None:
            super().__init__()
            self.thinker = resolve_qwen_thinker(qwen_model)
            self.audio_tower = self.thinker.audio_tower
            self.token_embedding = resolve_qwen_token_embedding(self.thinker)
            for module in (self.audio_tower, self.token_embedding):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                module.eval()
            if not all(hasattr(self.audio_tower, name) for name in ("ln_post", "proj1", "proj2")):
                raise AttributeError("Qwen audio tower must expose ln_post/proj1/proj2")
            self.pre_projector_dim = int(self.audio_tower.proj1.in_features)
            self.post_projector_dim = int(self.audio_tower.proj2.out_features)
            if self.pre_projector_dim != 896 or self.post_projector_dim != 1024:
                raise ValueError(
                    "GLCLAP v1 requires Qwen3-ASR-0.6B dimensions 896->1024; "
                    f"found {self.pre_projector_dim}->{self.post_projector_dim}"
                )

        def train(self, mode: bool = True) -> "QwenGLCLAPEncoder":
            """Keep all frozen Qwen modules in evaluation mode."""

            super().train(mode)
            self.audio_tower.eval()
            self.token_embedding.eval()
            return self

        def extract_audio_features(
            self,
            input_features: Any,
            feature_lens: Any,
        ) -> QwenAudioFeatures:
            """Return detached ``[T,896]`` and ``[T,1024]`` features."""

            captured: dict[str, Any] = {}
            timing: dict[str, float] = {}
            started = 0.0

            def capture_pre(_module: Any, _inputs: Any, output: Any) -> None:
                _sync(output)
                captured["pre"] = output.detach()
                timing["aut_ms"] = (time.perf_counter() - started) * 1000.0

            handle = self.audio_tower.ln_post.register_forward_hook(capture_pre)
            try:
                _sync(input_features)
                started = time.perf_counter()
                with torch.no_grad():
                    output = self.audio_tower(input_features, feature_lens=feature_lens)
                post = _last_hidden(output).detach()
                _sync(post)
            finally:
                handle.remove()
            if "pre" not in captured:
                raise RuntimeError("Qwen ln_post hook did not observe audio features")
            total_ms = (time.perf_counter() - started) * 1000.0
            qwen_projector_ms = max(0.0, total_ms - timing.get("aut_ms", total_ms))
            timing["qwen_projector_ms"] = qwen_projector_ms
            timing["projector_ms"] = qwen_projector_ms
            pre = captured["pre"]
            if pre.ndim == 3 and pre.shape[0] == 1:
                pre = pre[0]
            if post.ndim == 3 and post.shape[0] == 1:
                post = post[0]
            if pre.ndim != 2 or post.ndim != 2:
                raise ValueError("Qwen audio features must resolve to [T, D]")
            return QwenAudioFeatures(pre, post, timing)

        def embed_text(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
            """Mean-pool frozen Qwen token embeddings to ``[B, 1024]``."""

            with torch.no_grad():
                token_states = self.token_embedding(input_ids)
                if attention_mask is None:
                    pooled = token_states.mean(dim=1)
                else:
                    mask = attention_mask.to(token_states.dtype).unsqueeze(-1)
                    pooled = (token_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return pooled.detach()


    class GLCLAPMLPAdapter(nn.Module):
        """LayerNorm-MLP projection followed by unit normalization."""

        def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, output_dim: int = 512) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(input_dim)
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.activation = nn.GELU()
            self.linear2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, states: Any) -> Any:
            """Project ``[..., input_dim]`` states to normalized embeddings."""

            states = states.to(self.linear1.weight.dtype)
            output = self.linear2(self.activation(self.linear1(self.norm(states))))
            return F.normalize(output.float(), p=2, dim=-1)


    class GLCLAPAdapters(nn.Module):
        """Trainable 1024-to-512 audio and text adapter pair."""

        def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, output_dim: int = 512) -> None:
            super().__init__()
            self.audio = GLCLAPMLPAdapter(input_dim, hidden_dim, output_dim)
            self.text = GLCLAPMLPAdapter(input_dim, hidden_dim, output_dim)


    class RetrievalAudioProjector(nn.Module):
        """Independent 896-to-1024 Qwen-shaped retrieval projector."""

        def __init__(self, proj1: Any, proj2: Any, *, random_init: bool = False) -> None:
            super().__init__()
            self.proj1 = copy.deepcopy(proj1)
            self.activation = nn.GELU()
            self.proj2 = copy.deepcopy(proj2)
            if random_init:
                self.proj1.reset_parameters()
                self.proj2.reset_parameters()

        @classmethod
        def from_audio_tower(cls, audio_tower: Any, *, random_init: bool = False) -> "RetrievalAudioProjector":
            """Clone Qwen projector weights or create a shape-matched random copy."""

            return cls(audio_tower.proj1, audio_tower.proj2, random_init=random_init)

        def forward(self, states: Any) -> Any:
            """Map pre-projector ``[T,896]`` states to ``[T,1024]``."""

            states = states.to(self.proj1.weight.dtype)
            return self.proj2(self.activation(self.proj1(states)))


    class GLCLAPRetrieverModel(nn.Module):
        """Frozen Qwen feature source plus trainable GLCLAP retrieval branch."""

        MODES = {
            "qwen_post_projector_frozen",
            "qwen_projector_warmstart",
            "random_projector",
        }

        def __init__(
            self,
            encoder: QwenGLCLAPEncoder,
            mode: str = "qwen_post_projector_frozen",
            embedding_dim: int = 512,
            temperature: float = 0.07,
        ) -> None:
            super().__init__()
            if mode not in self.MODES:
                raise ValueError(f"unsupported GLCLAP mode: {mode}")
            if temperature <= 0:
                raise ValueError("temperature must be positive")
            self.encoder = encoder
            self.mode = mode
            self.embedding_dim = int(embedding_dim)
            self.adapters = GLCLAPAdapters(1024, 1024, self.embedding_dim)
            if mode == "qwen_post_projector_frozen":
                self.retrieval_projector = None
            else:
                self.retrieval_projector = RetrievalAudioProjector.from_audio_tower(
                    encoder.audio_tower,
                    random_init=mode == "random_projector",
                ).float()
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature), dtype=torch.float32))
            self.set_epoch(0)

        @property
        def temperature(self) -> Any:
            """Return a positive, numerically bounded trainable temperature."""

            return self.log_temperature.exp().clamp(1e-3, 1.0)

        def set_epoch(self, epoch: int) -> None:
            """Apply the projector freeze schedule for initialization ablations."""

            if self.retrieval_projector is not None:
                trainable = int(epoch) >= 1
                for parameter in self.retrieval_projector.parameters():
                    parameter.requires_grad_(trainable)

        def encode_audio_features(self, features: QwenAudioFeatures) -> Any:
            """Map one Qwen feature sequence to normalized ``[T,512]`` frames."""

            if self.retrieval_projector is None:
                source = features.post_projector.detach()
            else:
                source = self.retrieval_projector(features.pre_projector.detach())
            return self.adapters.audio(source)

        def encode_text_tokens(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
            """Map frozen mean-pooled token embeddings to normalized keys."""

            return self.adapters.text(self.encoder.embed_text(input_ids, attention_mask))

        def forward(
            self,
            audio_features: Sequence[QwenAudioFeatures],
            transcript_input_ids: Any,
            transcript_attention_mask: Any,
            hotword_input_ids: Any,
            hotword_attention_mask: Any,
        ) -> tuple[list[Any], Any, Any, Any]:
            """Encode a training batch through the DDP-visible retrieval branch.

            Frozen Qwen audio/text feature extraction happens before this call;
            all trainable adapters, the optional retrieval projector, and the
            temperature remain registered on this module for gradient sync.
            """

            audio_sequences = [self.encode_audio_features(item) for item in audio_features]
            transcript_keys = self.encode_text_tokens(
                transcript_input_ids, transcript_attention_mask
            )
            hotword_keys = self.encode_text_tokens(hotword_input_ids, hotword_attention_mask)
            return audio_sequences, transcript_keys, hotword_keys, self.temperature

        def optimizer_parameter_groups(
            self,
            adapter_lr: float = 3e-4,
            projector_lr: float = 3e-5,
        ) -> list[dict[str, Any]]:
            """Return distinct adapter/temperature and projector LR groups."""

            groups: list[dict[str, Any]] = [
                {
                    "name": "adapters",
                    "params": [*self.adapters.parameters(), self.log_temperature],
                    "lr": float(adapter_lr),
                }
            ]
            if self.retrieval_projector is not None:
                groups.append(
                    {
                        "name": "retrieval_projector",
                        # Include frozen epoch-0 parameters so unfreezing does
                        # not require rebuilding/resuming the optimizer.
                        "params": list(self.retrieval_projector.parameters()),
                        "lr": float(projector_lr),
                    }
                )
            return groups

        def retrieval_state_dict(self) -> dict[str, Any]:
            """Return only small trainable retrieval parameters, excluding Qwen."""

            state: dict[str, Any] = {
                f"adapters.{key}": value.detach().cpu()
                for key, value in self.adapters.state_dict().items()
            }
            state["log_temperature"] = self.log_temperature.detach().cpu()
            if self.retrieval_projector is not None:
                state.update(
                    {
                        f"retrieval_projector.{key}": value.detach().cpu()
                        for key, value in self.retrieval_projector.state_dict().items()
                    }
                )
            return state

        def load_retrieval_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
            """Load a small retrieval-only state dictionary."""

            adapter_state = {
                key.removeprefix("adapters."): value
                for key, value in state.items()
                if key.startswith("adapters.")
            }
            self.adapters.load_state_dict(adapter_state, strict=strict)
            if "log_temperature" in state:
                self.log_temperature.data.copy_(state["log_temperature"].to(self.log_temperature.device))
            elif strict:
                raise KeyError("checkpoint lacks log_temperature")
            projector_state = {
                key.removeprefix("retrieval_projector."): value
                for key, value in state.items()
                if key.startswith("retrieval_projector.")
            }
            if self.retrieval_projector is None:
                if strict and projector_state:
                    raise ValueError("checkpoint contains a projector for frozen-projector mode")
            else:
                self.retrieval_projector.load_state_dict(projector_state, strict=strict)


    class QwenGLCLAPRuntime:
        """PCM/text preprocessing facade used by index and streaming CLIs."""

        def __init__(self, model: GLCLAPRetrieverModel, processor: Any, sample_rate: int = 16000) -> None:
            self.model = model
            self.processor = processor
            self.sample_rate = int(sample_rate)
            first = next(model.adapters.parameters())
            self.device = first.device
            self.dtype = next(model.encoder.audio_tower.parameters()).dtype

        def encode_pcm(self, pcm16k: Sequence[float]) -> AudioEncoding:
            """Encode accumulated mono PCM into normalized GLCLAP frames."""

            if not pcm16k:
                return AudioEncoding(np.empty((0, self.model.embedding_dim), dtype=np.float32))
            waveform = np.asarray(pcm16k, dtype=np.float32)
            feature_batch = self.processor.feature_extractor(
                [waveform],
                sampling_rate=self.sample_rate,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_features = feature_batch["input_features"][0].to(self.device, dtype=self.dtype)
            attention_mask = feature_batch["attention_mask"][0].to(self.device)
            feature_len = attention_mask.sum().long().unsqueeze(0)
            qwen_features = self.model.encoder.extract_audio_features(input_features, feature_len)
            with torch.inference_mode():
                if self.model.retrieval_projector is None:
                    source = qwen_features.post_projector.detach()
                    retrieval_projector_ms = None
                else:
                    _sync(qwen_features.pre_projector)
                    projector_started = time.perf_counter()
                    source = self.model.retrieval_projector(qwen_features.pre_projector.detach())
                    _sync(source)
                    retrieval_projector_ms = (time.perf_counter() - projector_started) * 1000.0
                adapter_started = time.perf_counter()
                frames = self.model.adapters.audio(source)
            _sync(frames)
            adapter_ms = (time.perf_counter() - adapter_started) * 1000.0
            timings = dict(qwen_features.timings_ms)
            if retrieval_projector_ms is not None:
                timings["retrieval_projector_ms"] = retrieval_projector_ms
                timings["projector_ms"] = retrieval_projector_ms
            timings["adapter_ms"] = adapter_ms
            return AudioEncoding(frames.detach().float(), timings)

        def tokenize_texts(self, texts: Sequence[str]) -> tuple[Any, Any]:
            """Tokenize text variants without generation/chat special tokens."""

            if not texts:
                return (
                    torch.empty((0, 0), dtype=torch.long, device=self.device),
                    torch.empty((0, 0), dtype=torch.long, device=self.device),
                )
            tokens = self.processor.tokenizer(
                list(texts),
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            return tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device)

        def encode_texts(self, texts: Sequence[str], batch_size: int = 512) -> np.ndarray:
            """Encode text variants to fp32 unit-normalized retrieval keys."""

            outputs: list[np.ndarray] = []
            self.model.eval()
            for begin in range(0, len(texts), max(1, int(batch_size))):
                input_ids, attention_mask = self.tokenize_texts(texts[begin : begin + batch_size])
                with torch.inference_mode():
                    keys = self.model.encode_text_tokens(input_ids, attention_mask)
                outputs.append(keys.detach().float().cpu().numpy())
            if not outputs:
                return np.empty((0, self.model.embedding_dim), dtype=np.float32)
            return np.concatenate(outputs, axis=0)


else:

    class _TorchRequired:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("GLCLAP model components require PyTorch; install the 'probe' extra")


    QwenGLCLAPEncoder = _TorchRequired  # type: ignore[misc,assignment]
    GLCLAPMLPAdapter = _TorchRequired  # type: ignore[misc,assignment]
    GLCLAPAdapters = _TorchRequired  # type: ignore[misc,assignment]
    RetrievalAudioProjector = _TorchRequired  # type: ignore[misc,assignment]
    GLCLAPRetrieverModel = _TorchRequired  # type: ignore[misc,assignment]
    QwenGLCLAPRuntime = _TorchRequired  # type: ignore[misc,assignment]


def save_glclap_checkpoint(
    path: str | Path,
    model: Any,
    *,
    config: Mapping[str, Any],
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    epoch: int = -1,
    global_step: int = 0,
    training_state: Mapping[str, Any] | None = None,
) -> None:
    """Save retrieval-only parameters and reproducibility state."""

    if torch is None:
        raise ImportError("checkpointing requires PyTorch")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": "glclap-checkpoint-v1",
        "mode": model.mode,
        "embedding_dim": model.embedding_dim,
        "retrieval_state": model.retrieval_state_dict(),
        "config": dict(config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "training_state": dict(training_state or {}),
    }
    torch.save(payload, target)


def load_glclap_checkpoint(
    path: str | Path,
    qwen_model: Any,
    *,
    map_location: str | Any = "cpu",
) -> tuple[Any, Mapping[str, Any]]:
    """Construct a retrieval model and load a retrieval-only checkpoint."""

    if torch is None:
        raise ImportError("checkpoint loading requires PyTorch")
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("format_version") != "glclap-checkpoint-v1":
        raise ValueError("unsupported GLCLAP checkpoint format")
    encoder = QwenGLCLAPEncoder(qwen_model)
    model = GLCLAPRetrieverModel(
        encoder,
        mode=str(payload["mode"]),
        embedding_dim=int(payload.get("embedding_dim", 512)),
    )
    model.load_retrieval_state_dict(payload["retrieval_state"])
    return model, payload
