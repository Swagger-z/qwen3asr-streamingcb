"""End-to-end smoke test for the GLCLAP catalog preparation CLI."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.build_glclap_catalogs import main


class CatalogPreparationCLITests(unittest.TestCase):
    def test_builds_training_evaluation_and_report_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hkust = root / "hkust.txt"
            magicdata = root / "magicdata.txt"
            targets = root / "targets.jsonl"
            manifest = root / "test.jsonl"
            negative_output = root / "negative.jsonl"
            evaluation_output = root / "evaluation.jsonl"
            report = root / "report.json"
            hkust.write_text(
                "就是 10\n参观 9\n那边 8\n上面 7\n今天 6\n明天 5\n",
                encoding="utf-8",
            )
            magicdata.write_text(
                "欢迎 10\n来到 9\n大学 8\n公司 7\n学校 6\n老师 5\n",
                encoding="utf-8",
            )
            targets.write_text(
                json.dumps(
                    {"catalog_version": "targets-v1", "id": "ne-1", "text": "人工智能岛"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "utt_id": "u1",
                        "audio": "/data/u1.wav",
                        "text": "今天参观人工智能岛",
                        "target_hotword_ids": ["ne-1"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "build_glclap_catalogs",
                "--word-freq",
                f"hkust={hkust}",
                "--word-freq",
                f"magicdata={magicdata}",
                "--negative-output",
                str(negative_output),
                "--target-catalog",
                str(targets),
                "--eval-manifest",
                str(manifest),
                "--evaluation-output",
                str(evaluation_output),
                "--report",
                str(report),
                "--size",
                "4",
                "--seed",
                "42",
            ]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            negative_records = [json.loads(line) for line in negative_output.read_text(encoding="utf-8").splitlines()]
            evaluation_records = [json.loads(line) for line in evaluation_output.read_text(encoding="utf-8").splitlines()]
            report_record = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(len(negative_records), 4)
        self.assertEqual(len(evaluation_records), 4)
        self.assertEqual(evaluation_records[0]["id"], "ne-1")
        self.assertNotIn("今天", {record["text"] for record in evaluation_records[1:]})
        self.assertNotIn("参观", {record["text"] for record in evaluation_records[1:]})
        self.assertEqual(report_record["negative_records"], 4)
        self.assertEqual(report_record["evaluation_records"], 4)
        self.assertEqual(len(report_record["negative_sha256"]), 64)
        self.assertEqual(len(report_record["evaluation_sha256"]), 64)
        self.assertTrue(all(len(source["sha256"]) == 64 for source in report_record["sources"]))


if __name__ == "__main__":
    unittest.main()
