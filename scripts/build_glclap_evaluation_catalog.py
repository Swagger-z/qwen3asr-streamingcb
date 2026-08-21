"""Build a target-labelled evaluation catalog with clean distractors."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import (
    build_evaluation_catalog,
    load_jsonl,
    merge_word_frequency_sources,
    sha256_file,
    target_spellings,
    transcript_texts,
    validate_manifest_targets,
    write_jsonl,
)


def _word_frequency_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SOURCE=PATH")
    source, path = (part.strip() for part in value.split("=", 1))
    if not source or not path:
        raise argparse.ArgumentTypeError("SOURCE and PATH must both be non-empty")
    return source, path


def main() -> None:
    """Build an evaluation-only target+distractor vocabulary and provenance report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word-freq", action="append", required=True, type=_word_frequency_source)
    parser.add_argument("--target-catalog", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=8)
    parser.add_argument("--version", default="aishell1-ne-10k-v2")
    args = parser.parse_args()
    if args.size <= 0 or args.min_chars <= 0 or args.max_chars < args.min_chars:
        raise ValueError("invalid size or character range")

    targets = load_jsonl(args.target_catalog)
    manifest = load_jsonl(args.eval_manifest)
    validate_manifest_targets(manifest, targets)
    entries, source_reports = merge_word_frequency_sources(
        args.word_freq,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    records = build_evaluation_catalog(
        targets,
        entries,
        size=args.size,
        evaluation_transcripts=transcript_texts(manifest),
        seed=args.seed,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        version=args.version,
    )
    count = write_jsonl(args.output, records)
    report = {
        "format_version": 1,
        "purpose": "evaluation_target_and_distractor_catalog",
        "used_for_training": False,
        "seed": args.seed,
        "requested_size": args.size,
        "target_records": len(targets),
        "target_spellings": len(target_spellings(targets)),
        "evaluation_manifest_records": len(manifest),
        "merged_unique_distractor_terms": len(entries),
        "output": str(Path(args.output)),
        "records": count,
        "sha256": sha256_file(args.output),
        "sources": [asdict(item) for item in source_reports],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
