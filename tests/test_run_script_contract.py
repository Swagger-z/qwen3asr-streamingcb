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


if __name__ == "__main__":
    unittest.main()
