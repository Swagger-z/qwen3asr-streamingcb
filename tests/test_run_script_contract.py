"""Static contract tests for the end-to-end Linux experiment runner."""

from __future__ import annotations

import unittest
from pathlib import Path


class RunScriptContractTests(unittest.TestCase):
    def test_all_numbered_stages_and_core_commands_are_present(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "run.sh").read_text(encoding="utf-8")
        self.assertTrue(script.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("set -Eeuo pipefail", script)
        for stage in range(10):
            self.assertIn(f"stage{stage}()", script)
            self.assertIn(f"{stage}|stage{stage})", script)
        for command in (
            "build_glclap_training_pool.py",
            "prepare_aishell_ner.py",
            "build_glclap_evaluation_catalog.py",
            "build_boundary_stress.py",
            "train_glclap_retriever.py",
            "build_glclap_index.py",
            "decode_streaming_retrieval.py",
            "eval_hotword_retrieval.py",
        ):
            self.assertIn(command, script)

    def test_dataset_and_output_paths_are_environment_overridable(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "run.sh").read_text(encoding="utf-8")
        for variable in (
            "HKUST_WORD_FREQ",
            "MAGICDATA_WORD_FREQ",
            "AISHELL1_TRAIN_MANIFEST",
            "AISHELL_NER_DEV_ANNOTATED_TRANSCRIPT",
            "AISHELL_NER_DEV_WAV_ROOT",
            "AISHELL_NER_DEV_TARGET_CATALOG",
            "AISHELL_NER_DEV_EVAL_MANIFEST",
            "AISHELL_NER_DEV_ENTITY_MANIFEST",
            "AISHELL_NER_DEV_PREPARATION_REPORT",
            "AISHELL_NER_ANNOTATED_TRANSCRIPT",
            "AISHELL_NER_WAV_ROOT",
            "AISHELL_NER_TARGET_CATALOG",
            "AISHELL_NER_EVAL_MANIFEST",
            "AISHELL_NER_ENTITY_MANIFEST",
            "AISHELL_NER_ALIGNED_MANIFEST",
            "QWEN_MODEL",
            "DATA_ROOT",
            "OUTPUT_ROOT",
        ):
            self.assertIn(f'{variable}="${{{variable}:-', script)

    def test_training_requires_validation_and_uses_best_checkpoint(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "run.sh").read_text(encoding="utf-8")
        self.assertIn('--dev-manifest "${AISHELL_NER_DEV_ENTITY_MANIFEST}"', script)
        self.assertIn("--split dev", script)
        self.assertIn('evaluation.strategy=${EVAL_STRATEGY}', script)
        self.assertIn('evaluation.steps=${EVAL_STEPS}', script)
        self.assertIn('checkpoint="${output_dir}/best.pt"', script)

    def test_multi_gpu_training_uses_torchrun_and_global_batch(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "run.sh").read_text(encoding="utf-8")
        self.assertIn('NUM_GPUS="${NUM_GPUS:-${VISIBLE_GPU_COUNT}}"', script)
        self.assertIn("-m torch.distributed.run --standalone", script)
        self.assertIn('training.global_batch_size=${GLOBAL_BATCH_SIZE}', script)
        self.assertIn('"--nproc_per_node=${NUM_GPUS}"', script)


if __name__ == "__main__":
    unittest.main()
