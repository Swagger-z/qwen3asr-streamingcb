"""Dependency-light helpers for GLCLAP distributed training."""

from __future__ import annotations

import math
from typing import Sequence, TypeVar


T = TypeVar("T")


def resolve_gradient_accumulation(
    *,
    global_batch_size: int,
    micro_batch_size: int,
    world_size: int,
) -> int:
    """Derive per-rank accumulation while preserving the global batch size."""

    if global_batch_size <= 0 or micro_batch_size <= 0 or world_size <= 0:
        raise ValueError("global batch, micro batch, and world size must be positive")
    per_update = micro_batch_size * world_size
    if global_batch_size % per_update:
        raise ValueError(
            "training.global_batch_size must be divisible by "
            "training.micro_batch_size * WORLD_SIZE; "
            f"got {global_batch_size} % ({micro_batch_size} * {world_size})"
        )
    return global_batch_size // per_update


def shard_epoch_records(records: Sequence[T], *, world_size: int, rank: int) -> list[T]:
    """Return an equal-length deterministic rank shard, padding like DistributedSampler."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    values = list(records)
    if not values:
        return []
    samples_per_rank = math.ceil(len(values) / world_size)
    total_size = samples_per_rank * world_size
    padding = total_size - len(values)
    if padding:
        repeats = math.ceil(padding / len(values))
        values.extend((values * repeats)[:padding])
    return values[rank:total_size:world_size]


def shard_evaluation_records(records: Sequence[T], *, world_size: int, rank: int) -> list[T]:
    """Return a non-padding validation shard so every example is scored once."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    return list(records)[rank::world_size]
