"""Merge evaluation JSONL manifests while preserving records and gold labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import sha256_file, write_jsonl
from asr.contextual.glclap_runtime import jsonl_records
from asr.data.manifest import manifest_key, manifest_language


def main() -> None:
    """Concatenate manifests, reject duplicate keys, and report composition."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    records = []
    seen = set()
    languages: Counter[str] = Counter()
    inputs = []
    for value in args.input:
        path = Path(value).resolve()
        rows = jsonl_records(path)
        for record in rows:
            key = manifest_key(record)
            if key in seen:
                raise ValueError(f"duplicate evaluation manifest key: {key}")
            seen.add(key)
            languages[manifest_language(record, default="zh")] += 1
            records.append(record)
        inputs.append({"path": str(path), "sha256": sha256_file(path), "records": len(rows)})
    write_jsonl(args.output, records)
    report = {
        "format_version": 1,
        "purpose": "merged_evaluation_manifest",
        "records": len(records),
        "language_counts": dict(sorted(languages.items())),
        "inputs": inputs,
        "output_sha256": sha256_file(args.output),
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
