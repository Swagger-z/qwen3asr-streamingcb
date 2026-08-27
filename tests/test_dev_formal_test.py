"""CPU contracts for the formal dev runner and explicit offline retrieval."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

from asr.contextual.glclap import AudioEncoding, HotwordEmbeddingIndex
from asr.eval.glclap_metrics import evaluate_glclap_records
from scripts.decode_streaming_retrieval import _index_metadata, _retrieve_waveform, _validate_targets


ROOT = Path(__file__).resolve().parents[1]


class CountingEncoder:
    """Deterministic [N] PCM to [1,2] feature stub without torch/Qwen."""

    def __init__(self):
        self.calls = []

    def encode_pcm(self, waveform):
        """Record submitted lengths and return deterministic frame features."""
        self.calls.append(len(waveform))
        return AudioEncoding(np.array([[1.0, len(waveform) / 16000]], dtype=np.float32))


class DevFormalRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = HotwordEmbeddingIndex(np.eye(2, dtype=np.float32), ["a", "b"], ["A", "B"])

    def test_offline_is_single_pass_and_matches_streaming_tail(self):
        waveform = [0.0] * 84800
        offline_encoder, streaming_encoder = CountingEncoder(), CountingEncoder()
        offline = _retrieve_waveform(offline_encoder, self.index, waveform, mode="offline",
                                     chunk_size_sec=2.0, feed_samples=1600, top_k=2)
        streaming = _retrieve_waveform(streaming_encoder, self.index, waveform, mode="streaming",
                                       chunk_size_sec=2.0, feed_samples=1600, top_k=2)
        self.assertEqual(offline_encoder.calls, [84800])
        self.assertEqual(streaming_encoder.calls, [32000, 64000, 84800])
        self.assertEqual(offline[0].hits, streaming[-1].hits)
        self.assertTrue(offline[0].is_final)
        self.assertEqual(offline[0].accumulated_audio_sec, 5.3)
        self.assertEqual(len(offline), 1)

    def test_exact_chunk_finish_does_not_decode_twice(self):
        encoder = CountingEncoder()
        batches = _retrieve_waveform(encoder, self.index, [0.0] * 32000, mode="streaming",
                                     chunk_size_sec=2.0, feed_samples=1600, top_k=2)
        self.assertEqual(encoder.calls, [32000])
        self.assertEqual(len(batches), 1)
        self.assertTrue(batches[0].is_final)

    def test_checkpoint_index_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"best checkpoint")
            index_path = str(Path(directory) / "index.npz")
            with self.assertRaises(FileNotFoundError):
                _index_metadata(index_path, str(checkpoint), True)
            self.assertEqual(_index_metadata(index_path, str(checkpoint), False), {})
            metadata = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            Path(index_path + ".json").write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(_index_metadata(index_path, str(checkpoint), True), metadata)
            checkpoint.write_bytes(b"last checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                _index_metadata(index_path, str(checkpoint), True)

    def test_gold_coverage_and_unique_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.npz"
            self.index.save(path)
            valid = {"key": "u1", "target_hotword_ids": ["a", "b"]}
            _validate_targets([valid], str(path))
            for records, message in (([], "empty"), ([valid, valid], "duplicate"),
                                     ([{"key": "u1"}], "nonempty"),
                                     ([{"key": "u1", "target_hotword_ids": ["missing"]}], "absent")):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    _validate_targets(records, str(path))

    def test_multi_entity_hit_is_not_recall(self):
        metrics = evaluate_glclap_records([
            {"target_hotword_ids": ["a", "b", "c"], "final_hits": [{"hotword_id": "a"}]}
        ], ks=(1,))
        self.assertEqual(metrics["hit_at_1"], 1.0)
        self.assertAlmostEqual(metrics["recall_at_1"], 1 / 3)
        self.assertEqual(metrics["precision_at_1"], 1.0)
        self.assertEqual(metrics["target_entity_count"], 3)
        self.assertEqual(metrics["evaluable_utterances"], 1)


class DevShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Use Git Bash on Windows instead of the WSL launcher."""
        if os.name == "nt":
            candidate = Path("C:/Program Files/Git/bin/bash.exe")
            cls.bash = str(candidate) if candidate.is_file() else None
        else:
            cls.bash = shutil.which("bash")

    def run_shell(self, *arguments, **environment):
        """Run the script with isolated overrides without model/data access."""
        if not self.bash:
            self.skipTest("bash is unavailable")
        env = dict(os.environ)
        env.update({"DRY_RUN": "1", "CHECKPOINTS": "best last", "MODES": "offline streaming"})
        env.update(environment)
        return subprocess.run([self.bash, "run_dev_test.sh", *arguments], cwd=ROOT,
                              env=env, capture_output=True, text=True, encoding="utf-8")

    def test_syntax_and_full_dry_run(self):
        if not self.bash:
            self.skipTest("bash is unavailable")
        syntax = subprocess.run([self.bash, "-n", "run_dev_test.sh"], cwd=ROOT, capture_output=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / "not-created"
            result = self.run_shell("all", RUN_DIR=result_dir.as_posix())
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(result_dir.exists())
        self.assertEqual(result.stdout.count("scripts/decode_streaming_retrieval.py"), 4)
        self.assertEqual(result.stdout.count("scripts/build_glclap_index.py"), 2)
        self.assertEqual(result.stdout.count("--verify-offline"), 2)
        self.assertIn("--require-target-coverage", result.stdout)
        self.assertIn("--split dev", result.stdout)
        self.assertNotIn("build_boundary_stress.py", result.stdout)

    def test_stage_selection_custom_paths_and_modes(self):
        result = self.run_shell("stage3", EXP_DIR="/data/my experiment", CHECKPOINTS="best", MODES="offline")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("scripts/decode_streaming_retrieval.py"), 1)
        self.assertIn("--mode offline", result.stdout)
        self.assertIn("--checkpoint /data/my\\ experiment/best.pt", result.stdout)
        self.assertNotIn("--verify-offline", result.stdout)

    def test_invalid_stage_and_protected_output_fail(self):
        self.assertNotEqual(self.run_shell("stage9").returncode, 0)
        self.assertNotEqual(self.run_shell("stage3", CHECKPOINTS="typo").returncode, 0)
        self.assertNotEqual(self.run_shell("stage3", TOP_K="5").returncode, 0)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "best").mkdir()
            (run_dir / "best/offline.jsonl").write_text("{}\n", encoding="utf-8")
            (run_dir / "best/offline_metrics.json").write_text("{}\n", encoding="utf-8")
            result = self.run_shell("stage4", DRY_RUN="0", RUN_DIR=run_dir.as_posix(),
                                    CHECKPOINTS="best", MODES="offline")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("[exists]", result.stderr)

    def test_real_preparation_catalog_and_evaluation_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            wav_root = work / "wav"
            wav_root.mkdir()
            for key in ("u1", "u2"):
                with wave.open(str(wav_root / f"{key}.wav"), "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    audio.writeframes(b"\x00\x00" * 160)
            annotated = work / "dev.txt"
            annotated.write_text("u1 <北京大学>和<清华大学>\nu2 今天参观校园\n", encoding="utf-8")
            frequency = work / "word_freq.txt"
            frequency.write_text(
                "".join(f"词{chr(0x4E00 + index)} {100 - index}\n" for index in range(60)),
                encoding="utf-8",
            )
            run_dir = work / "results"
            env = dict(DRY_RUN="0", RUN_DIR=run_dir.as_posix(), CHECKPOINTS="best", MODES="offline",
                       PYTHON_BIN=Path(sys.executable).as_posix(), PYTHONIOENCODING="utf-8",
                       DEV_ANNOTATED_TRANSCRIPT=annotated.as_posix(), DEV_WAV_ROOT=wav_root.as_posix(),
                       HKUST_WORD_FREQ=frequency.as_posix(), MAGICDATA_WORD_FREQ=frequency.as_posix(),
                       CATALOG_SIZE="50")
            prepared = self.run_shell("stage0", "stage1", **env)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            records = [json.loads(line) for line in (run_dir / "data/dev_entities.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(len(records[0]["target_hotword_ids"]), 2)
            catalog = [json.loads(line) for line in (run_dir / "data/aishell_ner_dev_50.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(len(catalog), 50)
            self.assertTrue(set(records[0]["target_hotword_ids"]) <= {row["id"] for row in catalog})
            (run_dir / "best").mkdir()
            result = {**records[0], "final_hits": [{"hotword_id": records[0]["target_hotword_ids"][0]}]}
            (run_dir / "best/offline.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
            evaluated = self.run_shell("stage4", **env)
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            metrics = json.loads((run_dir / "best/offline_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["hit_at_1"], 1.0)
            self.assertEqual(metrics["recall_at_1"], 0.5)


if __name__ == "__main__":
    unittest.main()
