"""Adaptive rollback and candidate lifecycle tests."""

from __future__ import annotations

import unittest

from asr.contextual.candidates import CandidateManager
from asr.contextual.catalog import HotwordCatalog
from asr.contextual.commit import CommitController, CommitInvariantError
from asr.contextual.types import CandidateEvidence, CandidateStatus, HotwordEntry, RetrievalUpdate


class CommitTests(unittest.TestCase):
    def test_adaptive_formula_and_maximum_rollback(self) -> None:
        controller = CommitController(base_rollback_tokens=5, max_rollback_tokens=32, unfixed_chunk_num=0)
        controller.commit_after_decode(list(range(15)), rollback_start=10, has_active_candidate=False, chunk_id=0)
        self.assertEqual(len(controller.committed_ids), 10)
        self.assertEqual(controller.rollback_start(50, [20], chunk_id=1), 20)
        self.assertEqual(controller.rollback_start(50, [0], chunk_id=1), 18)
        self.assertEqual(controller.rollback_start(50, [], chunk_id=1), 45)
    def test_short_baseline_keeps_one_token_like_official_qwen(self) -> None:
        controller = CommitController(base_rollback_tokens=5, max_rollback_tokens=32, unfixed_chunk_num=0)
        self.assertEqual(controller.rollback_start(3, [], chunk_id=0), 1)
        self.assertEqual(controller.rollback_start(0, [], chunk_id=0), 0)


    def test_committed_prefix_cannot_be_rewritten(self) -> None:
        controller = CommitController(2, 4, unfixed_chunk_num=0)
        controller.commit_after_decode([1, 2, 3, 4], 2, False, 0)
        with self.assertRaises(CommitInvariantError):
            controller.commit_after_decode([9, 2, 3, 4], 2, False, 1)

    def test_candidate_ttl_releases_hold(self) -> None:
        catalog = HotwordCatalog([HotwordEntry(id="long", text="longword")])
        manager = CandidateManager(catalog, enter_threshold=0.5, exit_threshold=0.3, active_ttl_chunks=3)
        manager.apply(
            RetrievalUpdate(active=(CandidateEvidence("long", 0.8, anchor_token=1),)),
            chunk_id=0,
            default_anchor_token=4,
        )
        self.assertEqual(manager.holding_candidates()[0].status, CandidateStatus.ACTIVE)
        manager.apply(RetrievalUpdate(), chunk_id=2, default_anchor_token=8)
        self.assertTrue(manager.holding_candidates())
        manager.apply(RetrievalUpdate(), chunk_id=3, default_anchor_token=9)
        self.assertFalse(manager.holding_candidates())


if __name__ == "__main__":
    unittest.main()
