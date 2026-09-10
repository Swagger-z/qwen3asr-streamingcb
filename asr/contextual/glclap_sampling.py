"""Deterministic corpus-proportional scheduling for GLCLAP training."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence


def _largest_remainder_quotas(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("sample count must be non-negative")
    population = sum(sizes.values())
    if population <= 0:
        if total:
            raise ValueError("cannot sample from empty corpus groups")
        return {name: 0 for name in sizes}
    exact = {name: total * size / population for name, size in sizes.items()}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(sizes, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def proportional_epoch_indices(
    indices_by_corpus: Mapping[str, Sequence[int]],
    *,
    seed: int,
    epoch: int,
    samples_per_epoch: int | None = None,
) -> list[int]:
    """Return a reproducible raw-size-proportional global epoch schedule.

    ``samples_per_epoch`` unset or zero means a full pass where every record is
    used exactly once. A positive smaller value allocates exact per-corpus
    quotas with the largest-remainder method and samples without replacement.
    """

    groups = {str(name): list(values) for name, values in indices_by_corpus.items()}
    if not groups or not any(groups.values()):
        return []
    if any(not values for values in groups.values()):
        raise ValueError("corpus groups must not be empty")
    all_values = [value for values in groups.values() for value in values]
    if len(all_values) != len(set(all_values)):
        raise ValueError("corpus index groups must be disjoint")
    population = len(all_values)
    requested = population if not samples_per_epoch else int(samples_per_epoch)
    if requested <= 0:
        raise ValueError("samples_per_epoch must be positive or unset")
    if requested > population:
        raise ValueError(
            f"samples_per_epoch={requested} exceeds the {population} unique records"
        )
    if len(groups) == 1 and requested == population:
        result = list(all_values)
        random.Random(int(seed) + int(epoch)).shuffle(result)
        return result
    sizes = {name: len(values) for name, values in groups.items()}
    quotas = _largest_remainder_quotas(sizes, requested)
    selected: list[int] = []
    for name in sorted(groups):
        values = list(groups[name])
        random.Random(f"{int(seed)}:{int(epoch)}:{name}").shuffle(values)
        selected.extend(values[: quotas[name]])
    random.Random(f"{int(seed)}:{int(epoch)}:interleave").shuffle(selected)
    return selected


def scheduled_corpus_counts(
    schedule: Sequence[int], corpus_by_index: Sequence[str]
) -> dict[str, int]:
    """Count scheduled rows per corpus for logging and protocol checks."""

    counts: dict[str, int] = {}
    for index in schedule:
        corpus = str(corpus_by_index[index])
        counts[corpus] = counts.get(corpus, 0) + 1
    return dict(sorted(counts.items()))
