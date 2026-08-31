"""Deterministic sampling helpers for the Chinese GLCLAP training loop."""

from __future__ import annotations

import hashlib
import random
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

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


def batch_negative_exclusions(
    transcripts: Iterable[str],
    *,
    min_chars: int = 2,
    max_chars: int = 8,
) -> set[str]:
    """Return spoken terms that must not be sampled as batch negatives.

    The full normalized transcript is included alongside every contiguous
    substring in the local-positive length range. This prevents a term that is
    genuinely present in the audio from becoming a false negative merely
    because another local span was sampled for the current epoch.
    """

    if min_chars <= 0 or max_chars < min_chars:
        raise ValueError("invalid negative-exclusion length range")
    excluded: set[str] = set()
    for transcript in transcripts:
        normalized = compact_transcript(transcript)
        if not normalized:
            continue
        excluded.add(normalized)
        upper = min(max_chars, len(normalized))
        for length in range(min_chars, upper + 1):
            for start in range(0, len(normalized) - length + 1):
                excluded.add(normalized[start : start + length])
    return excluded


def annotated_entity_positives(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deduplicated gold entity texts from an annotated manifest row.

    The validation path intentionally consumes ``entities[].text`` produced by
    :mod:`scripts.prepare_aishell_ner`; it never infers a hotword from the plain
    transcript or from the negative catalog.
    """

    raw_entities = record.get("entities")
    if not isinstance(raw_entities, Sequence) or isinstance(
        raw_entities, (str, bytes)
    ):
        raise ValueError("annotated validation record requires an entities list")
    positives: list[str] = []
    seen: set[str] = set()
    transcript = compact_transcript(str(record.get("target", record.get("text", ""))))
    for index, raw_entity in enumerate(raw_entities):
        if not isinstance(raw_entity, Mapping):
            raise ValueError(f"entities[{index}] must be an object")
        text = compact_transcript(str(raw_entity.get("text", "")))
        if not text:
            raise ValueError(f"entities[{index}] is missing non-empty text")
        if transcript and text not in transcript:
            raise ValueError(
                f"entities[{index}] text {text!r} is absent from the transcript"
            )
        if text not in seen:
            seen.add(text)
            positives.append(text)
    if not positives:
        raise ValueError("annotated validation record contains no gold entities")
    return tuple(positives)


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


class SharedNegativeSampler:
    """Pre-canonicalized shared-negative sampler without per-step sorting."""

    def __init__(self, vocabulary: Sequence[str]) -> None:
        self.vocabulary = tuple(sorted({item for item in vocabulary if item}))

    def sample(
        self,
        positives: Iterable[str],
        count: int = 4095,
        *,
        seed: int = 42,
        epoch: int = 0,
        step: int = 0,
        strict: bool = True,
    ) -> tuple[str, ...]:
        """Sample from the canonical vocabulary while excluding positives."""

        excluded = set(positives)
        available = [item for item in self.vocabulary if item not in excluded]
        if strict and len(available) < count:
            raise ValueError(
                f"need {count} unique negatives after exclusion, found {len(available)}"
            )
        count = min(count, len(available))
        generator = random.Random((int(seed) << 32) ^ (int(epoch) << 16) ^ int(step))
        return tuple(generator.sample(available, count))


def equality_positive_mask(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    """Build a multi-positive equality mask for duplicate-aware contrastive loss."""

    return np.asarray([[a == b for b in right] for a in left], dtype=np.bool_)


def transcript_positive_mask(
    transcripts: Sequence[str], candidates: Sequence[str]
) -> np.ndarray:
    """Return a ``[B, K]`` mask of candidates spoken in each transcript.

    Match nonempty contiguous substrings after the same NFKC/whitespace
    normalization as local-positive sampling. This includes positives supplied
    by *other* batch rows, not just the span sampled for the current row.
    Candidate order, duplicate columns and empty ``[B, 0]`` shapes are retained.
    This training-only helper does not infer validation/test entity labels.
    """

    normalized_transcripts = [compact_transcript(text) for text in transcripts]
    normalized_candidates = [compact_transcript(text) for text in candidates]
    mask = np.zeros((len(transcripts), len(candidates)), dtype=np.bool_)
    for row, transcript in enumerate(normalized_transcripts):
        for column, candidate in enumerate(normalized_candidates):
            mask[row, column] = bool(candidate) and candidate in transcript
    return mask


def membership_positive_mask(
    left: Sequence[Iterable[str]], right: Sequence[str]
) -> np.ndarray:
    """Build a multi-label positive mask for one or more gold terms per audio."""

    groups = [set(values) for values in left]
    if any(not values for values in groups):
        raise ValueError("every local retrieval row requires at least one positive")
    return np.asarray(
        [[candidate in values for candidate in right] for values in groups],
        dtype=np.bool_,
    )
