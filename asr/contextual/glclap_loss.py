"""Global/local bidirectional contrastive objectives used by GLCLAP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - optional training dependency.
    torch = None
    F = None


@dataclass(frozen=True)
class GLCLAPLossOutput:
    """Scalar loss components and similarity matrices."""

    loss: Any
    global_loss: Any
    local_loss: Any
    global_logits: Any
    local_logits: Any


def multi_positive_contrastive_loss(logits: Any, positive_mask: Any) -> Any:
    """Compute row/column InfoNCE with one or more positives per anchor.

    Rows or columns without positives are omitted from the corresponding
    direction.  This supports duplicate transcripts/hotwords in a batch while
    ensuring they are never treated as negatives.
    """

    if torch is None:
        raise ImportError("GLCLAP loss requires PyTorch; install the 'probe' extra")
    if logits.ndim != 2 or positive_mask.shape != logits.shape:
        raise ValueError("logits and positive_mask must have equal [N, M] shape")
    mask = positive_mask.bool()

    def direction(values: Any, positives: Any) -> Any:
        valid = positives.any(dim=1)
        if not bool(valid.any()):
            return values.sum() * 0.0
        selected = values[valid]
        selected_mask = positives[valid]
        positive_values = selected.masked_fill(~selected_mask, float("-inf"))
        numerator = torch.logsumexp(positive_values, dim=1)
        denominator = torch.logsumexp(selected, dim=1)
        return (denominator - numerator).mean()

    return 0.5 * (direction(logits, mask) + direction(logits.T, mask.T))


def max_over_time_similarity(audio_frames: Any, text_keys: Any, frame_mask: Any | None = None) -> Any:
    """Return ``[B,K]`` similarities using GLCLAP max-over-time pooling."""

    if torch is None:
        raise ImportError("GLCLAP loss requires PyTorch; install the 'probe' extra")
    if audio_frames.ndim != 3 or text_keys.ndim != 2:
        raise ValueError("audio_frames must be [B,T,D] and text_keys [K,D]")
    similarities = torch.einsum("btd,kd->btk", audio_frames, text_keys)
    if frame_mask is not None:
        if frame_mask.shape != audio_frames.shape[:2]:
            raise ValueError("frame_mask must have shape [B,T]")
        similarities = similarities.masked_fill(~frame_mask.bool().unsqueeze(-1), float("-inf"))
    return similarities.max(dim=1).values


def glclap_loss(
    audio_frames: Any,
    transcript_keys: Any,
    hotword_keys: Any,
    global_positive_mask: Any,
    local_positive_mask: Any,
    *,
    temperature: Any,
    frame_mask: Any | None = None,
    global_weight: float = 1.0,
    local_weight: float = 1.0,
) -> GLCLAPLossOutput:
    """Compute weighted global and local bidirectional contrastive losses."""

    if torch is None:
        raise ImportError("GLCLAP loss requires PyTorch; install the 'probe' extra")
    if audio_frames.ndim != 3:
        raise ValueError("audio_frames must have shape [B,T,D]")
    if frame_mask is None:
        pooled_audio = audio_frames.mean(dim=1)
    else:
        mask = frame_mask.to(audio_frames.dtype).unsqueeze(-1)
        pooled_audio = (audio_frames * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    pooled_audio = F.normalize(pooled_audio, p=2, dim=-1)
    transcript_keys = F.normalize(transcript_keys, p=2, dim=-1)
    hotword_keys = F.normalize(hotword_keys, p=2, dim=-1)
    scale = 1.0 / torch.as_tensor(temperature, device=audio_frames.device).clamp_min(1e-3)
    global_logits = (pooled_audio @ transcript_keys.T) * scale
    local_logits = max_over_time_similarity(audio_frames, hotword_keys, frame_mask) * scale
    global_value = multi_positive_contrastive_loss(global_logits, global_positive_mask)
    local_value = multi_positive_contrastive_loss(local_logits, local_positive_mask)
    total = float(global_weight) * global_value + float(local_weight) * local_value
    return GLCLAPLossOutput(total, global_value, local_value, global_logits, local_logits)
