import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SplitCatalogCLITests(unittest.TestCase):
    def test_training_pool_does_not_read_evaluation_annotations(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            word_freq = work / "word_freq.txt"
            terms = ["测试目标", "训练词语", "语音识别", "人工智能", "上下文", "检索系统"]
            word_freq.write_text(
                "".join(f"{term} {100-index}\n" for index, term in enumerate(terms)),
                encoding="utf-8",
            )
            output = work / "training.jsonl"
            report = work / "training.report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "build_glclap_training_pool.py"),
                    "--word-freq",
                    f"toy={word_freq}",
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                    "--size",
                    "4",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            metadata = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(metadata["uses_evaluation_annotations"])
            self.assertEqual(metadata["records"], 4)

    def test_evaluation_catalog_contains_targets_and_unspoken_distractors(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            word_freq = work / "word_freq.txt"
            word_freq.write_text(
                "训练词语 100\n语音识别 90\n人工智能 80\n上下文本 70\n检索系统 60\n",
                encoding="utf-8",
            )
            target = work / "targets.jsonl"
            target.write_text(
                json.dumps(
                    {
                        "catalog_version": "target-v1",
                        "id": "ne-1",
                        "text": "商汤科技",
                        "aliases": ["商汤"],
                        "language": "zh",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = work / "eval.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "utt_id": "u1",
                        "audio": "/data/u1.wav",
                        "text": "参观商汤科技",
                        "target_hotword_ids": ["ne-1"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output = work / "evaluation.jsonl"
            report = work / "evaluation.report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "build_glclap_evaluation_catalog.py"),
                    "--word-freq",
                    f"toy={word_freq}",
                    "--target-catalog",
                    str(target),
                    "--eval-manifest",
                    str(manifest),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                    "--size",
                    "4",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["id"], "ne-1")
            self.assertEqual(len(records), 4)
            self.assertFalse(json.loads(report.read_text(encoding="utf-8"))["used_for_training"])


if __name__ == "__main__":
    unittest.main()
