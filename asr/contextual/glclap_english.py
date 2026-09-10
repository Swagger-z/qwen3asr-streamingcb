"""English n-gram statistics and deterministic synthetic hotword labels."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .glclap_data import english_tokens, normalize_english_term


def normalized_english_tokens(text: str) -> tuple[str, ...]:
    """Return case-folded word tokens used for English statistics."""

    return tuple(token.casefold() for token in english_tokens(text))


def count_english_ngrams(
    transcripts: Iterable[str], *, max_words: int = 4
) -> tuple[Counter[str], Counter[str]]:
    """Count unigrams and all contiguous terms up to *max_words*."""

    if max_words <= 0:
        raise ValueError("max_words must be positive")
    unigram_counts: Counter[str] = Counter()
    ngram_counts: Counter[str] = Counter()
    for transcript in transcripts:
        tokens = normalized_english_tokens(transcript)
        unigram_counts.update(tokens)
        for width in range(1, min(max_words, len(tokens)) + 1):
            ngram_counts.update(
                " ".join(tokens[start : start + width])
                for start in range(0, len(tokens) - width + 1)
            )
    return unigram_counts, ngram_counts


def common_unigrams(counts: Mapping[str, int], limit: int = 5000) -> set[str]:
    """Return the deterministically ranked most frequent unigram spellings."""

    if limit < 0:
        raise ValueError("common unigram limit must be non-negative")
    ranked = sorted(counts, key=lambda token: (-int(counts[token]), token))
    return set(ranked[:limit])


def select_synthetic_phrases(
    transcript: str,
    *,
    training_unigram_counts: Mapping[str, int],
    excluded_common: set[str],
    utt_id: str,
    max_targets: int = 3,
    max_words: int = 4,
    seed: int = 42,
    allowed_terms: set[str] | None = None,
) -> tuple[str, ...]:
    """Choose stable rare-containing whole-word spans from one transcript."""

    if max_targets <= 0 or max_words <= 0:
        raise ValueError("max_targets and max_words must be positive")
    surface = english_tokens(transcript)
    normalized = tuple(token.casefold() for token in surface)
    candidates: dict[str, tuple[tuple[Any, ...], str]] = {}
    for width in range(1, min(max_words, len(surface)) + 1):
        for start in range(0, len(surface) - width + 1):
            norm_span = normalized[start : start + width]
            if not any(token not in excluded_common for token in norm_span):
                continue
            text = " ".join(surface[start : start + width])
            key = normalize_english_term(text)
            if allowed_terms is not None and key not in allowed_terms:
                continue
            rarest = min(int(training_unigram_counts.get(token, 0)) for token in norm_span)
            digest = hashlib.sha256(
                f"{seed}:{utt_id}:{start}:{width}:{key}".encode("utf-8")
            ).hexdigest()
            score = (math.log1p(rarest), -width, digest)
            current = candidates.get(key)
            if current is None or score < current[0]:
                candidates[key] = (score, text)
    ranked = sorted(candidates.values(), key=lambda item: item[0])
    selected: list[str] = []
    selected_normalized: list[tuple[str, ...]] = []
    for _score, text in ranked:
        tokens = tuple(normalize_english_term(text).split())
        # Avoid redundant nested labels from the same rare occurrence.
        if any(tokens == existing for existing in selected_normalized):
            continue
        selected.append(text)
        selected_normalized.append(tokens)
        if len(selected) == max_targets:
            break
    return tuple(selected)


def stable_english_id(prefix: str, text: str) -> str:
    """Return a content-addressed ID that preserves English word boundaries."""

    normalized = normalize_english_term(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"
