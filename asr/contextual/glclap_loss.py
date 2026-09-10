"""Global/local bidirectional contrastive objectives used by GLCLAP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


@dataclass(frozen=True)
class MultilingualGLCLAPLossOutput:
    """Loss components for a mixed batch with language-local candidate sets."""

    loss: Any
    global_loss: Any
    local_loss: Any
    global_logits: Any
    local_logits: Mapping[str, Any]


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


def mean_pool_audio(audio_frames: Any, frame_mask: Any | None = None) -> Any:
    """Mean-pool ``[B,T,D]`` audio frames into normalized ``[B,D]`` keys."""

    if torch is None:
        raise ImportError("GLCLAP loss requires PyTorch; install the 'probe' extra")
    if audio_frames.ndim != 3:
        raise ValueError("audio_frames must have shape [B,T,D]")
    if frame_mask is None:
        pooled = audio_frames.mean(dim=1)
    else:
        if frame_mask.shape != audio_frames.shape[:2]:
            raise ValueError("frame_mask must have shape [B,T]")
        mask = frame_mask.to(audio_frames.dtype).unsqueeze(-1)
        pooled = (audio_frames * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled, p=2, dim=-1)


def global_contrastive_loss(
    audio_frames: Any,
    transcript_keys: Any,
    positive_mask: Any,
    *,
    temperature: Any,
    frame_mask: Any | None = None,
) -> tuple[Any, Any]:
    """Return global transcript loss and ``[B,B]`` logits."""

    pooled_audio = mean_pool_audio(audio_frames, frame_mask)
    transcript_keys = F.normalize(transcript_keys, p=2, dim=-1)
    scale = 1.0 / torch.as_tensor(temperature, device=audio_frames.device).clamp_min(1e-3)
    logits = (pooled_audio @ transcript_keys.T) * scale
    return multi_positive_contrastive_loss(logits, positive_mask), logits


def local_contrastive_loss(
    audio_frames: Any,
    hotword_keys: Any,
    positive_mask: Any,
    *,
    temperature: Any,
    frame_mask: Any | None = None,
) -> tuple[Any, Any]:
    """Return max-over-time local loss and ``[B,K]`` logits."""

    if torch is None:
        raise ImportError("GLCLAP loss requires PyTorch; install the 'probe' extra")
    hotword_keys = F.normalize(hotword_keys, p=2, dim=-1)
    scale = 1.0 / torch.as_tensor(temperature, device=audio_frames.device).clamp_min(1e-3)
    logits = max_over_time_similarity(audio_frames, hotword_keys, frame_mask) * scale
    return multi_positive_contrastive_loss(logits, positive_mask), logits


def multilingual_glclap_loss(
    audio_frames: Any,
    transcript_keys: Any,
    language_hotword_keys: Mapping[str, Any],
    global_positive_mask: Any,
    local_positive_masks: Mapping[str, Any],
    language_rows: Mapping[str, Sequence[int] | Any],
    *,
    temperature: Any,
    frame_mask: Any | None = None,
    global_weight: float = 1.0,
    local_weight: float = 1.0,
    compute_local: bool = True,
) -> MultilingualGLCLAPLossOutput:
    """Compute one global loss plus row-weighted same-language local losses."""

    if torch is None:
        raise ImportError("GLCLAP loss requires PyTorch; install the 'probe' extra")
    global_value, global_logits = global_contrastive_loss(
        audio_frames,
        transcript_keys,
        global_positive_mask,
        temperature=temperature,
        frame_mask=frame_mask,
    )
    local_logits: dict[str, Any] = {}
    local_value = audio_frames.sum() * 0.0
    if compute_local:
        if set(language_hotword_keys) != set(local_positive_masks) or set(
            language_hotword_keys
        ) != set(language_rows):
            raise ValueError("multilingual local-loss language groups must match")
        total_rows = sum(len(rows) for rows in language_rows.values())
        if total_rows != audio_frames.shape[0]:
            raise ValueError("language row groups must cover every audio row exactly once")
        flattened = [int(row) for rows in language_rows.values() for row in rows]
        if sorted(flattened) != list(range(audio_frames.shape[0])):
            raise ValueError("language row groups must be disjoint and exhaustive")
        for language in sorted(language_rows):
            rows = torch.as_tensor(
                language_rows[language], dtype=torch.long, device=audio_frames.device
            )
            selected_mask = frame_mask.index_select(0, rows) if frame_mask is not None else None
            value, logits = local_contrastive_loss(
                audio_frames.index_select(0, rows),
                language_hotword_keys[language],
                local_positive_masks[language],
                temperature=temperature,
                frame_mask=selected_mask,
            )
            local_logits[language] = logits
            local_value = local_value + value * (len(rows) / max(1, total_rows))
    total = float(global_weight) * global_value + float(local_weight) * local_value
    return MultilingualGLCLAPLossOutput(
        total, global_value, local_value, global_logits, local_logits
    )


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
    compute_local: bool = True,
) -> GLCLAPLossOutput:
    """Compute weighted global and local bidirectional contrastive losses."""

    if torch is None:
        raise ImportError("GLCLAP loss requires PyTorch; install the 'probe' extra")
    global_value, global_logits = global_contrastive_loss(
        audio_frames,
        transcript_keys,
        global_positive_mask,
        temperature=temperature,
        frame_mask=frame_mask,
    )
    if not compute_local:
        local_logits = audio_frames.new_empty((audio_frames.shape[0], hotword_keys.shape[0]))
        local_value = audio_frames.sum() * 0.0
    else:
        local_value, local_logits = local_contrastive_loss(
            audio_frames,
            hotword_keys,
            local_positive_mask,
            temperature=temperature,
            frame_mask=frame_mask,
        )
    total = float(global_weight) * global_value + float(local_weight) * local_value
    return GLCLAPLossOutput(total, global_value, local_value, global_logits, local_logits)
