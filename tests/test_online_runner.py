"""Runner contracts and optional real GPU replay parity."""

import os
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OnlineRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Use native Git Bash on Windows, normal Bash on Linux."""
        cls.bash = ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))
        if not cls.bash or not Path(cls.bash).is_file():
            raise unittest.SkipTest("bash unavailable")

    def run_script(self, *stages, **overrides):
        env = dict(os.environ)
        env.update(DRY_RUN="1", CHECKPOINTS="best last", REPLAY_MODES="fast realtime",
                   DATASETS="original boundary", RUN_ALIGNER="0", INDEX_PATH="")
        env.update(overrides)
        return subprocess.run([self.bash, "run_online_eval.sh", *stages], cwd=ROOT,
                              env=env, capture_output=True, text=True, encoding="utf-8")

    def test_syntax_full_dry_run_and_no_writes(self):
        syntax = subprocess.run([self.bash, "-n", "run_online_eval.sh"], cwd=ROOT, capture_output=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "not-created"
            result = self.run_script("all", RUN_DIR=run_dir.as_posix())
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(run_dir.exists())
        self.assertEqual(result.stdout.count("scripts/decode_streaming_retrieval.py"), 8)
        self.assertEqual(result.stdout.count("scripts/eval_hotword_retrieval.py"), 8)
        self.assertEqual(result.stdout.count("scripts/build_glclap_index.py"), 2)
        self.assertIn("--require-entity-timestamps", result.stdout)
        self.assertIn("--warmup-refreshes 3", result.stdout)
        self.assertNotIn("train_glclap", result.stdout)

    def test_dev_custom_experiment_and_index_reuse(self):
        result = self.run_script("stage2", "stage3", SPLIT="dev", EXP_DIR="/data/my experiment",
                                 CHECKPOINTS="best", REPLAY_MODES="fast", DATASETS="original",
                                 INDEX_PATH="/data/old index.npz", CHUNK_MS="560")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dev.jsonl", result.stdout)
        self.assertIn("--checkpoint /data/my\\ experiment/best.pt", result.stdout)
        self.assertIn("--index /data/old\\ index.npz", result.stdout)
        self.assertIn("streaming.chunk_size_sec=0.560", result.stdout)
        self.assertNotIn("scripts/build_glclap_index.py", result.stdout)
        self.assertEqual(result.stdout.count("scripts/decode_streaming_retrieval.py"), 1)

    def test_aligner_is_opt_in_and_original_skips_boundary(self):
        result = self.run_script("stage0", "stage1", RUN_ALIGNER="1", DATASETS="original")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scripts/align_hotword_manifest.py", result.stdout)
        self.assertNotIn("scripts/build_boundary_stress.py", result.stdout)

    def test_invalid_configuration_and_existing_outputs_fail(self):
        for overrides in ({"TOP_K": "20"}, {"SPLIT": "train"}, {"CHUNK_MS": "0"},
                          {"CHECKPOINTS": "typo"}, {"REPLAY_MODES": "bad"},
                          {"DATASETS": "bad"}, {"DEADLINES_MS": "-1"},
                          {"INDEX_PATH": "/x.npz"}, {"WARMUP_REFRESHES": "-1"}):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(self.run_script("all", **overrides).returncode, 0)
        self.assertNotEqual(self.run_script("stage99").returncode, 0)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "best").mkdir()
            (work / "best/original_fast.jsonl").touch()
            (work / "best/original_fast_metrics.json").touch()
            result = self.run_script("stage4", RUN_DIR=work.as_posix(), DRY_RUN="0",
                                     CHECKPOINTS="best", REPLAY_MODES="fast", DATASETS="original")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("[exists]", result.stderr)

    def test_real_stage0_stage1_and_stage4_without_models(self):
        from test_online_retrieval import TimedDataTests, example_record
        from asr.data.timed_entities import validate_timed_records
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source, aligned = TimedDataTests().fixtures(work)
            source_path, aligned_path, catalog = work / "source.jsonl", work / "aligned.jsonl", work / "targets.jsonl"
            source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
            aligned_path.write_text("".join(json.dumps(r) + "\n" for r in aligned), encoding="utf-8")
            catalog.write_text('{"id":"gold","text":"gold"}\n', encoding="utf-8")
            run_dir = work / "run"
            env = dict(DRY_RUN="0", PYTHON_BIN=Path(sys.executable).as_posix(),
                       RUN_DIR=run_dir.as_posix(), SOURCE_MANIFEST=source_path.as_posix(),
                       ALIGNED_MANIFEST=aligned_path.as_posix(), TARGET_CATALOG=catalog.as_posix(),
                       CHECKPOINTS="best", REPLAY_MODES="fast")
            result = self.run_script("stage0", "stage1", **env)
            self.assertEqual(result.returncode, 0, result.stderr)
            variants = [json.loads(line) for line in (run_dir / "data/boundary.jsonl").read_text().splitlines()]
            self.assertEqual(len(variants), 14)
            validate_timed_records(variants, check_audio=True)
            grouped = json.loads((run_dir / "data/original_timed.jsonl").read_text())
            record = {**grouped, **example_record(entities=grouped["entities"]),
                      "source": source["source"], "audio": source["source"],
                      "audio_sha256": grouped["audio_sha256"]}
            (run_dir / "best").mkdir()
            (run_dir / "best/original_fast.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            result = self.run_script("stage4", **env, DATASETS="original", BOOTSTRAP_SAMPLES="20")
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((run_dir / "best/original_fast_metrics.json").read_text())
            self.assertAlmostEqual(metrics["online"]["by_k"]["50"]["latency_ms_mean"], 1900)


@unittest.skipUnless(os.environ.get("RUN_GLCLAP_GPU_REPLAY") == "1",
                     "set RUN_GLCLAP_GPU_REPLAY=1 with Qwen/checkpoint/index/WAV fixtures")
class GPUReplayParityTests(unittest.TestCase):
    def test_fast_and_real_gpu_replay_have_identical_prefixes_and_rankings(self):
        """Real hardware test, never substituted with a synthetic latency claim."""
        import torch
        from asr.audio_io import read_wav_mono_float
        from asr.config import load_config
        from asr.contextual.glclap import HotwordEmbeddingIndex
        from asr.contextual.glclap_runtime import build_glclap_runtime
        from scripts.decode_streaming_retrieval import _index_metadata, _retrieve_waveform

        self.assertTrue(torch.cuda.is_available(), "GPU replay test requires CUDA")
        config = load_config(os.environ.get("GLCLAP_CONFIG", "configs/glclap/qwen_post_projector_frozen.yaml"),
                             ["model.qwen_model=" + os.environ["QWEN_MODEL_PATH"]])
        checkpoint, index_path = os.environ["GLCLAP_CHECKPOINT"], os.environ["GLCLAP_INDEX"]
        _index_metadata(index_path, checkpoint, True)
        model, runtime, _, _ = build_glclap_runtime(config, checkpoint=checkpoint)
        model.eval()
        index = HotwordEmbeddingIndex.load(index_path).to(next(model.adapters.parameters()).device)
        audio = read_wav_mono_float(os.environ["QWEN_TEST_WAV"], 16000)[:round(2.1 * 16000)]
        self.assertGreater(len(audio), 16000, "fixture needs more than one second of PCM")

        class RecordingRuntime:
            def __init__(self):
                self.inputs = []

            def encode_pcm(self, waveform):
                self.inputs.append(tuple(waveform))
                return runtime.encode_pcm(waveform)

        for _ in range(3):
            encoded = runtime.encode_pcm(audio[:8960])
            index.search(encoded.frames, 50)
        del encoded
        all_batches, inputs = [], []
        for mode in ("fast", "realtime"):
            recorder = RecordingRuntime()
            batches = _retrieve_waveform(recorder, index, audio, mode="streaming",
                                         chunk_size_sec=.56, feed_samples=1600, top_k=50,
                                         replay_mode=mode)
            all_batches.append(batches)
            inputs.append(recorder.inputs)
            previous = 0
            for batch in batches:
                stamp = batch.timeline
                self.assertGreaterEqual(stamp["start_sec"] + 1e-7, max(previous, stamp["ready_sec"]))
                self.assertAlmostEqual(stamp["finish_sec"] - stamp["audio_cutoff_sec"],
                                       stamp["feed_wait_sec"] + stamp["queue_wait_sec"] +
                                       stamp["processing_sec"], places=6)
                previous = stamp["finish_sec"]
        self.assertEqual(inputs[0], inputs[1])
        self.assertEqual([[h.hotword_id for h in b.hits] for b in all_batches[0]],
                         [[h.hotword_id for h in b.hits] for b in all_batches[1]])


if __name__ == "__main__":
    unittest.main()
