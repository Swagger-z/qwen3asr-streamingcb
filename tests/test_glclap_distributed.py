"""Tests for GLCLAP distributed batch and sampler semantics."""

from __future__ import annotations

import unittest

from asr.contextual.glclap_distributed import (
    resolve_gradient_accumulation,
    shard_epoch_records,
    shard_evaluation_records,
)


class DistributedBatchTests(unittest.TestCase):
    def test_global_batch_384_is_preserved_across_gpu_counts(self) -> None:
        self.assertEqual(
            resolve_gradient_accumulation(
                global_batch_size=384,
                micro_batch_size=8,
                world_size=1,
            ),
            48,
        )
        self.assertEqual(
            resolve_gradient_accumulation(
                global_batch_size=384,
                micro_batch_size=8,
                world_size=4,
            ),
            12,
        )
        self.assertEqual(
            resolve_gradient_accumulation(
                global_batch_size=384,
                micro_batch_size=8,
                world_size=8,
            ),
            6,
        )

    def test_non_divisible_global_batch_fails_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            resolve_gradient_accumulation(
                global_batch_size=384,
                micro_batch_size=8,
                world_size=7,
            )

    def test_evaluation_shards_cover_each_example_exactly_once(self) -> None:
        records = list(range(10))
        shards = [
            shard_evaluation_records(records, world_size=4, rank=rank)
            for rank in range(4)
        ]
        self.assertEqual([len(shard) for shard in shards], [3, 3, 2, 2])
        self.assertEqual(sorted(item for shard in shards for item in shard), records)

    def test_rank_shards_have_equal_length_and_deterministic_padding(self) -> None:
        records = list(range(10))
        shards = [shard_epoch_records(records, world_size=4, rank=rank) for rank in range(4)]
        self.assertEqual([len(shard) for shard in shards], [3, 3, 3, 3])
        self.assertEqual(shards[0], [0, 4, 8])
        self.assertEqual(shards[2], [2, 6, 0])
        self.assertEqual(shards[3], [3, 7, 1])


if __name__ == "__main__":
    unittest.main()
