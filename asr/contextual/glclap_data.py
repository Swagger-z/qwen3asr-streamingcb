"""Deterministic sampling helpers for the Chinese GLCLAP training loop."""

from __future__ import annotations

import hashlib
import random
import unicodedata
from typing import Iterable, Sequence

import numpy as np


def compact_transcript(text: str) -> str:
    """NFKC-normalize a transcript and remove whitespace for substring sampling."""

    return "".join(unicodedata.normalize("NFKC", text).split())


def deterministic_local_positive(
    text: str,
    *,
    utt_id: str,
    epoch: int,
    seed: int = 42,
    min_chars: int = 2,
    max_chars: int = 8,
) -> str:
    """Sample one reproducible contiguous 2--8 character local positive."""

    normalized = compact_transcript(text)
    if not normalized:
        raise ValueError("cannot sample a local positive from an empty transcript")
    lower = min(min_chars, len(normalized))
    upper = min(max_chars, len(normalized))
    digest = hashlib.sha256(f"{seed}:{epoch}:{utt_id}".encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(digest[:8], "little"))
    length = generator.randint(lower, upper)
    start = generator.randint(0, len(normalized) - length)
    return normalized[start : start + length]


def sample_shared_negatives(
    vocabulary: Sequence[str],
    positives: Iterable[str],
    count: int = 4095,
    *,
    seed: int = 42,
    epoch: int = 0,
    step: int = 0,
    strict: bool = True,
) -> tuple[str, ...]:
    """Sample unique shared negatives while excluding every batch positive."""

    excluded = set(positives)
    available = sorted({item for item in vocabulary if item and item not in excluded})
    if strict and len(available) < count:
        raise ValueError(f"need {count} unique negatives after exclusion, found {len(available)}")
    count = min(count, len(available))
    generator = random.Random((int(seed) << 32) ^ (int(epoch) << 16) ^ int(step))
    return tuple(generator.sample(available, count))


def equality_positive_mask(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    """Build a multi-positive equality mask for duplicate-aware contrastive loss."""

    return np.asarray([[a == b for b in right] for a in left], dtype=np.bool_)
