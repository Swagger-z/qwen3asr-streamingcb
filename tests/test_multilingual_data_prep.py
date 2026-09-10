"""CLI-level tests for multilingual manifest and English label preparation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _run(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


class MultilingualPreparationTests(unittest.TestCase):
    def test_aishell2_converter_and_training_mix_namespace_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            wav_root = work / "wav"
            wav_root.mkdir()
            with wave.open(str(wav_root / "a2.wav"), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(b"\0\0" * 20)
            transcript = work / "transcript.txt"
            transcript.write_text("a2 中文转写\n", encoding="utf-8")
            aishell2 = work / "aishell2.jsonl"
            _run(
                "prepare_aishell2_manifest.py",
                "--transcript", str(transcript), "--wav-root", str(wav_root),
                "--output", str(aishell2), "--report", str(work / "a2.report.json"),
            )
            english = work / "english.jsonl"
            _jsonl(english, [{"key": "librispeech:e1", "source": "e1.flac", "target": "HELLO WORLD"}])
            mixed = work / "mixed.jsonl"
            _run(
                "build_glclap_training_mix.py",
                "--input", f"aishell2:zh:train={aishell2}",
                "--input", f"librispeech:en:train-clean-100={english}",
                "--output", str(mixed), "--report", str(work / "mix.report.json"),
            )
            rows = [json.loads(line) for line in mixed.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["key"] for row in rows], ["aishell2:a2", "librispeech:e1"])
            self.assertEqual([row["language"] for row in rows], ["zh", "en"])

    def test_english_pool_and_librispeech_labels_use_train_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            train = work / "train.jsonl"
            _jsonl(train, [
                {"key": "t1", "source": "t1.flac", "target": "THE ZEPHYR RETURNS"},
                {"key": "t2", "source": "t2.flac", "target": "THE QUANTUM HARBOR"},
                {"key": "t3", "source": "t3.flac", "target": "THE ZEPHYR HARBOR"},
            ])
            pool = work / "pool.jsonl"
            _run(
                "build_english_glclap_pool.py",
                "--manifest", str(train), "--output", str(pool),
                "--report", str(work / "pool.report.json"), "--size", "3",
                "--min-count", "1", "--exclude-common-unigrams", "1",
            )
            pool_rows = [json.loads(line) for line in pool.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(pool_rows), 3)
            self.assertTrue(all(row["language"] == "en" for row in pool_rows))
            evaluation = work / "eval.jsonl"
            _jsonl(evaluation, [{"key": "e1", "source": "e1.flac", "target": "THE ZEPHYR VISITED QUANTUM HARBOR"}])
            output = work / "eval.labels.jsonl"
            catalog = work / "targets.jsonl"
            _run(
                "prepare_librispeech_hotword_eval.py",
                "--train-manifest", str(train), "--eval-manifest", str(evaluation),
                "--split", "dev-clean", "--manifest-output", str(output),
                "--target-catalog-output", str(catalog), "--report", str(work / "eval.report.json"),
                "--exclude-common-unigrams", "1",
            )
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["language"], "en")
            self.assertGreaterEqual(len(record["entities"]), 1)
            self.assertEqual(record["target_hotword_ids"], [item["hotword_id"] for item in record["entities"]])

    def test_stop_without_transcript_is_explicitly_retrieval_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            audio = work / "u1.wav"
            audio.write_bytes(b"placeholder")
            (work / "wav.scp").write_text(f"u1 {audio}\n", encoding="utf-8")
            (work / "hotlists").write_text("u1 TAYLOR SWIFT\n", encoding="utf-8")
            (work / "hotlists.uniq").write_text("TAYLOR SWIFT\n", encoding="utf-8")
            manifest = work / "stop1.jsonl"
            _run(
                "prepare_stop_hotword_eval.py",
                "--wav-scp", str(work / "wav.scp"), "--hotlists", str(work / "hotlists"),
                "--hotlists-uniq", str(work / "hotlists.uniq"), "--dataset", "stop1",
                "--manifest-output", str(manifest), "--target-catalog-output", str(work / "targets.jsonl"),
                "--report", str(work / "report.json"),
            )
            record = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["target"], "")
            self.assertTrue(record["metadata"]["retrieval_only"])
            self.assertFalse(record["metadata"]["timing_evaluable"])

    def test_monolingual_catalog_size_and_bilingual_union_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            targets = work / "targets.jsonl"
            _jsonl(targets, [{"catalog_version": "targets", "id": "en:t", "text": "RARE WORD", "aliases": [], "language": "en"}])
            pool = work / "pool.jsonl"
            _jsonl(pool, [
                {"catalog_version": "pool", "id": "en:d1", "text": "BOSTON", "aliases": [], "language": "en"},
                {"catalog_version": "pool", "id": "en:d2", "text": "SEATTLE", "aliases": [], "language": "en"},
                {"catalog_version": "pool", "id": "en:d3", "text": "CHICAGO", "aliases": [], "language": "en"},
            ])
            manifest = work / "eval.jsonl"
            _jsonl(manifest, [{"key": "en:1", "source": "1.wav", "target": "A RARE WORD", "target_hotword_ids": ["en:t"], "language": "en"}])
            english = work / "en3.jsonl"
            _run(
                "build_glclap_catalog_from_pool.py",
                "--target-catalog", str(targets), "--distractor-pool", str(pool),
                "--eval-manifest", str(manifest), "--language", "en", "--size", "3",
                "--version", "en3", "--output", str(english), "--report", str(work / "en3.report.json"),
            )
            en_rows = [json.loads(line) for line in english.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(en_rows), 3)
            self.assertIn("en:t", {row["id"] for row in en_rows})
            chinese = work / "zh3.jsonl"
            _jsonl(chinese, [
                {"catalog_version": "zh3", "id": f"zh:{index}", "text": f"词{index}", "aliases": [], "language": "zh"}
                for index in range(3)
            ])
            mixed = work / "mixed6.jsonl"
            _run(
                "merge_glclap_catalogs.py",
                "--input", str(chinese), "--input", str(english),
                "--output", str(mixed), "--report", str(work / "mixed.report.json"),
                "--version", "mixed6",
            )
            mixed_rows = [json.loads(line) for line in mixed.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(mixed_rows), 6)
            self.assertEqual(len({row["id"] for row in mixed_rows}), 6)


if __name__ == "__main__":
    unittest.main()
