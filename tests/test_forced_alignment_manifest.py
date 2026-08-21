import json
import tempfile
import unittest
import wave
from pathlib import Path

from asr.contextual.types import HotwordEntry
from asr.data.forced_alignment import (
    AlignmentItem,
    aligned_records_for_source,
    normalize_alignment_text,
    resolve_hotword_span,
)


class ForcedAlignmentTests(unittest.TestCase):
    def test_normalization_and_cross_item_alias_match(self):
        items = (
            AlignmentItem("前往", 0.1, 0.3),
            AlignmentItem("张", 0.3, 0.4),
            AlignmentItem("江", 0.4, 0.5),
            AlignmentItem("人工智能岛。", 0.5, 1.1),
        )
        span = resolve_hotword_span(items, ("张江人工智能岛",))
        self.assertEqual(normalize_alignment_text(" A-Ｂ。 "), "ab")
        self.assertEqual((span.start_time, span.end_time), (0.3, 1.1))

    def test_repeated_mention_is_explicitly_resolved(self):
        items = (
            AlignmentItem("商汤", 0.1, 0.4),
            AlignmentItem("和", 0.4, 0.5),
            AlignmentItem("商汤", 0.5, 0.8),
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_hotword_span(items, ("商汤",))
        self.assertEqual(resolve_hotword_span(items, ("商汤",), "last").start_time, 0.5)

    def test_multiple_targets_expand_to_single_target_records(self):
        entries = {
            "ne-1": HotwordEntry(id="ne-1", text="商汤"),
            "ne-2": HotwordEntry(id="ne-2", text="张江"),
        }
        record = {
            "utt_id": "u1",
            "audio": "/tmp/u1.wav",
            "text": "商汤位于张江",
            "target_hotword_ids": ["ne-1", "ne-2"],
        }
        items = (
            AlignmentItem("商汤", 0.1, 0.4),
            AlignmentItem("位于", 0.4, 0.6),
            AlignmentItem("张江", 0.6, 0.9),
        )
        outputs = aligned_records_for_source(record, entries, items)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0]["target_hotword_ids"], ["ne-1"])
        self.assertEqual(outputs[1]["target_hotword_ids"], ["ne-2"])
        self.assertNotEqual(outputs[0]["utt_id"], outputs[1]["utt_id"])


class AlignedManifestCLIContractTests(unittest.TestCase):
    def test_validation_cli_accepts_complete_pcm16_manifest(self):
        from scripts.validate_aligned_manifest import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "u1.wav"
            with wave.open(str(audio), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(b"\x00\x00" * 16000)
            catalog = root / "catalog.jsonl"
            catalog.write_text(
                json.dumps({"id": "ne-1", "text": "商汤"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            source = root / "source.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "utt_id": "u1",
                        "audio": str(audio),
                        "text": "商汤",
                        "target_hotword_ids": ["ne-1"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            aligned = root / "aligned.jsonl"
            aligned.write_text(
                json.dumps(
                    {
                        "utt_id": "u1",
                        "source_utt_id": "u1",
                        "audio": str(audio),
                        "text": "商汤",
                        "target_hotword_ids": ["ne-1"],
                        "aligned_hotword_id": "ne-1",
                        "hotword_start_sec": 0.1,
                        "hotword_end_sec": 0.5,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = root / "report.json"
            argv = [
                "validate_aligned_manifest.py",
                "--source-manifest",
                str(source),
                "--aligned-manifest",
                str(aligned),
                "--catalog",
                str(catalog),
                "--report",
                str(report),
            ]
            import sys

            previous = sys.argv
            try:
                sys.argv = argv
                main()
            finally:
                sys.argv = previous
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])


class AlignerRunScriptContractTests(unittest.TestCase):
    def test_stages_and_commands_are_present(self):
        script = (Path(__file__).resolve().parents[1] / "run_aligner.sh").read_text(encoding="utf-8")
        self.assertTrue(script.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("set -Eeuo pipefail", script)
        for stage in range(3):
            self.assertIn(f"stage{stage}()", script)
        self.assertIn("scripts/align_hotword_manifest.py", script)
        self.assertIn("scripts/validate_aligned_manifest.py", script)
        self.assertIn("AISHELL_NER_ALIGNED_MANIFEST", script)


if __name__ == "__main__":
    unittest.main()
