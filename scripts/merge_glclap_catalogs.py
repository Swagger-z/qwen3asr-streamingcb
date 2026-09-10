"""Merge disjoint monolingual catalogs into one versioned bilingual catalog."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import load_jsonl, sha256_file, write_jsonl


def main() -> None:
    """Rewrite catalog versions, reject ID collisions, and report composition."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--version", default="zh-en-20k-v1")
    args = parser.parse_args()
    records = []
    seen: dict[str, tuple[str, str]] = {}
    languages: Counter[str] = Counter()
    inputs = []
    for path_value in args.input:
        path = Path(path_value).resolve()
        source_records = load_jsonl(path)
        for record in source_records:
            hotword_id = str(record["id"])
            language = str(record.get("language", "")).casefold()
            if language not in {"zh", "en"}:
                raise ValueError(f"catalog record has invalid language: {language!r}")
            identity = (language, str(record.get("text", "")))
            if hotword_id in seen:
                if seen[hotword_id] != identity:
                    raise ValueError(f"conflicting duplicate hotword ID: {hotword_id}")
                continue
            seen[hotword_id] = identity
            languages[language] += 1
            records.append({**record, "catalog_version": args.version, "language": language})
        inputs.append({"path": str(path), "sha256": sha256_file(path), "records": len(source_records)})
    records.sort(key=lambda record: (record["language"], str(record["id"])))
    write_jsonl(args.output, records)
    report = {
        "format_version": 1,
        "purpose": "bilingual_evaluation_catalog",
        "version": args.version,
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
