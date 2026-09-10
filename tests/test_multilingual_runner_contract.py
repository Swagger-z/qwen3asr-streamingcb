"""Static contract for the staged multilingual experiment runner."""

from __future__ import annotations

import unittest
from pathlib import Path


class MultilingualRunnerContractTests(unittest.TestCase):
    def test_runner_covers_data_training_indexes_and_paired_catalog_eval(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "run_multilingual_glclap.sh").read_text(encoding="utf-8")
        self.assertTrue(script.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("set -Eeuo pipefail", script)
        for stage in range(9):
            self.assertIn(f"stage{stage}()", script)
        for name in (
            "prepare_aishell2_manifest.py",
            "build_glclap_training_mix.py",
            "build_english_glclap_pool.py",
            "prepare_librispeech_hotword_eval.py",
            "prepare_stop_hotword_eval.py",
            "multilingual_global_only.yaml",
            "multilingual_frozen.yaml",
            "--negative-catalog \"zh=",
            "--negative-catalog \"en=",
            "--monolingual-reference",
        ):
            self.assertIn(name, script)


if __name__ == "__main__":
    unittest.main()
