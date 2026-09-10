"""Deterministic, language-aware sampling helpers for GLCLAP training."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_ENGLISH_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def compact_transcript(text: str) -> str:
    """NFKC-normalize a transcript and remove whitespace for substring sampling."""

    return "".join(unicodedata.normalize("NFKC", text).split())


def english_tokens(text: str) -> tuple[str, ...]:
    """Return punctuation-trimmed English word tokens preserving surface case."""

    normalized = unicodedata.normalize("NFKC", str(text)).replace("’", "'")
    return tuple(match.group(0) for match in _ENGLISH_WORD.finditer(normalized))


def normalize_english_term(text: str) -> str:
    """Normalize an English word or phrase for exact token-span matching."""

    return " ".join(token.casefold() for token in english_tokens(text))


def normalize_term_for_language(text: str, language: str) -> str:
    """Return the canonical matching form for Chinese or English text."""

    language = str(language).strip().casefold()
    if language == "zh":
        return compact_transcript(text)
    if language == "en":
        return normalize_english_term(text)
    raise ValueError(f"unsupported GLCLAP language: {language!r}")


def _surface_term(text: str, language: str) -> str:
    if language == "zh":
        return compact_transcript(text)
    return " ".join(english_tokens(text))


def deterministic_local_positive(
    text: str,
    *,
    utt_id: str,
    epoch: int,
    seed: int = 42,
    min_chars: int = 2,
    max_chars: int = 8,
    language: str = "zh",
    min_words: int = 1,
    max_words: int = 4,
) -> str:
    """Sample one reproducible contiguous character or whole-word positive."""

    language = str(language).strip().casefold()
    units = list(english_tokens(text)) if language == "en" else list(compact_transcript(text))
    if language not in {"zh", "en"}:
        raise ValueError(f"unsupported GLCLAP language: {language!r}")
    if not units:
        raise ValueError("cannot sample a local positive from an empty transcript")
    configured_min = min_words if language == "en" else min_chars
    configured_max = max_words if language == "en" else max_chars
    if configured_min <= 0 or configured_max < configured_min:
        raise ValueError("invalid local-positive length range")
    lower = min(configured_min, len(units))
    upper = min(configured_max, len(units))
    digest = hashlib.sha256(f"{seed}:{epoch}:{utt_id}".encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(digest[:8], "little"))
    length = generator.randint(lower, upper)
    start = generator.randint(0, len(units) - length)
    selected = units[start : start + length]
    return " ".join(selected) if language == "en" else "".join(selected)


def batch_negative_exclusions(
    transcripts: Iterable[str],
    *,
    min_chars: int = 2,
    max_chars: int = 8,
    language: str = "zh",
    min_words: int = 1,
    max_words: int = 4,
) -> set[str]:
    """Return spoken terms that must not be sampled as batch negatives.

    The full normalized transcript is included alongside every contiguous
    substring in the local-positive length range. This prevents a term that is
    genuinely present in the audio from becoming a false negative merely
    because another local span was sampled for the current epoch.
    """

    language = str(language).strip().casefold()
    configured_min = min_words if language == "en" else min_chars
    configured_max = max_words if language == "en" else max_chars
    if language not in {"zh", "en"}:
        raise ValueError(f"unsupported GLCLAP language: {language!r}")
    if configured_min <= 0 or configured_max < configured_min:
        raise ValueError("invalid negative-exclusion length range")
    excluded: set[str] = set()
    for transcript in transcripts:
        units = (
            list(normalize_english_term(transcript).split())
            if language == "en"
            else list(compact_transcript(transcript))
        )
        if not units:
            continue
        excluded.add(" ".join(units) if language == "en" else "".join(units))
        upper = min(configured_max, len(units))
        for length in range(configured_min, upper + 1):
            for start in range(0, len(units) - length + 1):
                span = units[start : start + length]
                excluded.add(" ".join(span) if language == "en" else "".join(span))
    return excluded


def annotated_entity_positives(
    record: Mapping[str, Any], *, language: str = "zh"
) -> tuple[str, ...]:
    """Return deduplicated gold entity texts from an annotated manifest row.

    The validation path consumes explicit ``entities[].text`` annotations from
    AISHELL-NER, LibriSpeech synthetic labels, or STOP conversion; it never
    infers labels from the plain transcript or negative catalog at evaluation.
    """

    raw_entities = record.get("entities")
    if not isinstance(raw_entities, Sequence) or isinstance(
        raw_entities, (str, bytes)
    ):
        raise ValueError("annotated validation record requires an entities list")
    positives: list[str] = []
    seen: set[str] = set()
    transcript = normalize_term_for_language(
        str(record.get("target", record.get("text", ""))), language
    )
    for index, raw_entity in enumerate(raw_entities):
        if not isinstance(raw_entity, Mapping):
            raise ValueError(f"entities[{index}] must be an object")
        raw_text = str(raw_entity.get("text", ""))
        normalized_text = normalize_term_for_language(raw_text, language)
        text = _surface_term(raw_text, language)
        if not normalized_text:
            raise ValueError(f"entities[{index}] is missing non-empty text")
        if transcript and not _term_occurs(transcript, normalized_text, language):
            raise ValueError(
                f"entities[{index}] text {text!r} is absent from the transcript"
            )
        if normalized_text not in seen:
            seen.add(normalized_text)
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
    language: str = "zh",
) -> tuple[str, ...]:
    """Sample unique shared negatives while excluding every batch positive."""

    excluded = {
        normalize_term_for_language(item, language) for item in positives if item
    }
    canonical = {
        normalize_term_for_language(item, language): _surface_term(item, language)
        for item in vocabulary
        if normalize_term_for_language(item, language)
    }
    available = sorted(value for key, value in canonical.items() if key not in excluded)
    if strict and len(available) < count:
        raise ValueError(f"need {count} unique negatives after exclusion, found {len(available)}")
    count = min(count, len(available))
    generator = random.Random((int(seed) << 32) ^ (int(epoch) << 16) ^ int(step))
    return tuple(generator.sample(available, count))


class SharedNegativeSampler:
    """Pre-canonicalized shared-negative sampler without per-step sorting."""

    def __init__(self, vocabulary: Sequence[str], *, language: str = "zh") -> None:
        self.language = str(language).strip().casefold()
        canonical: dict[str, str] = {}
        for item in vocabulary:
            key = normalize_term_for_language(item, self.language)
            if key:
                canonical.setdefault(key, _surface_term(item, self.language))
        self._canonical = canonical
        self.vocabulary = tuple(sorted(canonical.values()))

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

        excluded = {
            normalize_term_for_language(item, self.language)
            for item in positives
            if item
        }
        available = [
            value for key, value in self._canonical.items() if key not in excluded
        ]
        available.sort()
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
    transcripts: Sequence[str], candidates: Sequence[str], *, language: str = "zh"
) -> np.ndarray:
    """Return a ``[B, K]`` mask of candidates spoken in each transcript.

    Match nonempty contiguous substrings after the same NFKC/whitespace
    normalization as local-positive sampling. This includes positives supplied
    by *other* batch rows, not just the span sampled for the current row.
    Candidate order, duplicate columns and empty ``[B, 0]`` shapes are retained.
    This training-only helper does not infer validation/test entity labels.
    """

    normalized_transcripts = [
        normalize_term_for_language(text, language) for text in transcripts
    ]
    normalized_candidates = [
        normalize_term_for_language(text, language) for text in candidates
    ]
    mask = np.zeros((len(transcripts), len(candidates)), dtype=np.bool_)
    for row, transcript in enumerate(normalized_transcripts):
        for column, candidate in enumerate(normalized_candidates):
            mask[row, column] = bool(candidate) and _term_occurs(
                transcript, candidate, language
            )
    return mask


def _term_occurs(transcript: str, candidate: str, language: str) -> bool:
    if not candidate:
        return False
    if language == "zh":
        return candidate in transcript
    transcript_tokens = transcript.split()
    candidate_tokens = candidate.split()
    width = len(candidate_tokens)
    return any(
        transcript_tokens[start : start + width] == candidate_tokens
        for start in range(0, len(transcript_tokens) - width + 1)
    )


def membership_positive_mask(
    left: Sequence[Iterable[str]], right: Sequence[str], *, language: str | None = None
) -> np.ndarray:
    """Build a multi-label positive mask for one or more gold terms per audio."""

    if language is None:
        groups = [set(values) for values in left]
        candidates = list(right)
    else:
        groups = [
            {normalize_term_for_language(value, language) for value in values}
            for values in left
        ]
        candidates = [normalize_term_for_language(value, language) for value in right]
    if any(not values for values in groups):
        raise ValueError("every local retrieval row requires at least one positive")
    return np.asarray(
        [[candidate in values for candidate in candidates] for values in groups],
        dtype=np.bool_,
    )
