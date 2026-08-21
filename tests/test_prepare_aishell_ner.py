import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

from scripts.prepare_aishell_ner import parse_annotated_transcript


class PrepareAISHELLNERTests(unittest.TestCase):
    def test_official_markers_preserve_types_spans_and_repeated_mentions(self):
        transcript, mentions = parse_annotated_transcript(
            "<中原地产>首席分析师[张大伟]在(北京)和(北京)发言"
        )
        self.assertEqual(transcript, "中原地产首席分析师张大伟在北京和北京发言")
        self.assertEqual([item.entity_type for item in mentions], ["ORG", "PER", "LOC", "LOC"])
        self.assertEqual([item.occurrence_index for item in mentions], [0, 0, 0, 1])
        for mention in mentions:
            self.assertEqual(transcript[mention.char_start : mention.char_end], mention.text)

    def test_converter_emits_all_and_entity_only_views_without_inference(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "aishell_ner_transcript.test.txt"
            source.write_text(
                "u1 <中原地产>首席分析师[张大伟]说\n"
                "u2 (北京)与(北京)联合发布\n"
                "u3 这句话没有实体\n",
                encoding="utf-8",
            )
            wav_root = work / "wav"
            wav_root.mkdir()
            for utt_id in ("u1", "u2", "u3"):
                with wave.open(str(wav_root / f"{utt_id}.wav"), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(16000)
                    writer.writeframes(b"\x00\x00" * 160)
            catalog = work / "targets.jsonl"
            full_manifest = work / "test.jsonl"
            entity_manifest = work / "test_entities.jsonl"
            report = work / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "prepare_aishell_ner.py"),
                    "--annotated-transcript",
                    str(source),
                    "--wav-root",
                    str(wav_root),
                    "--target-catalog-output",
                    str(catalog),
                    "--eval-manifest-output",
                    str(full_manifest),
                    "--entity-manifest-output",
                    str(entity_manifest),
                    "--report",
                    str(report),
                    "--split",
                    "test",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            targets = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines()]
            full = [json.loads(line) for line in full_manifest.read_text(encoding="utf-8").splitlines()]
            entity_only = [
                json.loads(line) for line in entity_manifest.read_text(encoding="utf-8").splitlines()
            ]
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(len(targets), 3)
            self.assertEqual(len(full), 3)
            self.assertEqual(len(entity_only), 2)
            self.assertEqual(full[2]["target_hotword_ids"], [])
            self.assertEqual(len(full[1]["entities"]), 2)
            self.assertEqual(full[1]["entities"][0]["hotword_id"], full[1]["entities"][1]["hotword_id"])
            self.assertFalse(payload["entity_labels_inferred"])
            self.assertEqual(payload["entity_mention_count_by_type"], {"LOC": 2, "ORG": 1, "PER": 1})

    def test_malformed_markers_fail(self):
        with self.assertRaisesRegex(ValueError, "unclosed"):
            parse_annotated_transcript("前往(北京")
        with self.assertRaisesRegex(ValueError, "unmatched"):
            parse_annotated_transcript("前往北京)")


if __name__ == "__main__":
    unittest.main()
