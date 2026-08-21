"""Build GLCLAP training-negative and evaluation hotword catalogs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import (
    build_evaluation_catalog,
    build_negative_catalog,
    load_jsonl,
    merge_word_frequency_sources,
    target_spellings,
    sha256_file,
    transcript_texts,
    validate_manifest_targets,
    write_jsonl,
)


def _word_frequency_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SOURCE=PATH")
    source, path = value.split("=", 1)
    source = source.strip()
    path = path.strip()
    if not source or not path:
        raise argparse.ArgumentTypeError("SOURCE and PATH must both be non-empty")
    return source, path


def parse_args() -> argparse.Namespace:
    """Parse catalog preparation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--word-freq",
        action="append",
        required=True,
        type=_word_frequency_source,
        metavar="SOURCE=PATH",
        help="Repeatable TERM COUNT source, e.g. hkust=/data/hkust/word_freq.txt",
    )
    parser.add_argument("--negative-output", required=True)
    parser.add_argument("--target-catalog", help="Annotated target-only JSONL catalog")
    parser.add_argument("--eval-manifest", help="Evaluation JSONL containing text and target IDs")
    parser.add_argument("--evaluation-output", help="Targets plus distractors JSONL")
    parser.add_argument("--report", help="JSON preparation report; defaults beside negative output")
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=8)
    parser.add_argument("--negative-version", default="zh-train-neg-v1")
    parser.add_argument("--evaluation-version", default="aishell1-ne-10k-v1")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.min_chars <= 0 or args.max_chars < args.min_chars:
        raise ValueError("invalid --min-chars/--max-chars range")
    if args.evaluation_output and (not args.target_catalog or not args.eval_manifest):
        raise ValueError(
            "--evaluation-output requires both --target-catalog and --eval-manifest"
        )
    if args.eval_manifest and not args.target_catalog:
        raise ValueError("--eval-manifest requires --target-catalog")


def main() -> None:
    """Build deterministic catalogs and write a provenance/statistics report."""

    args = parse_args()
    _validate_args(args)
    entries, source_reports = merge_word_frequency_sources(
        args.word_freq,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    targets: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    if args.target_catalog:
        targets = load_jsonl(args.target_catalog)
    if args.eval_manifest:
        manifest = load_jsonl(args.eval_manifest)
        validate_manifest_targets(manifest, targets)
    target_terms = target_spellings(targets)
    negatives = build_negative_catalog(
        entries,
        size=args.size,
        excluded_terms=target_terms,
        seed=args.seed,
        version=args.negative_version,
    )
    negative_count = write_jsonl(args.negative_output, negatives)
    negative_sha256 = sha256_file(args.negative_output)
    evaluation_count = 0
    evaluation_sha256 = None
    if args.evaluation_output:
        evaluation = build_evaluation_catalog(
            targets,
            entries,
            size=args.size,
            evaluation_transcripts=transcript_texts(manifest),
            seed=args.seed,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            version=args.evaluation_version,
        )
        evaluation_count = write_jsonl(args.evaluation_output, evaluation)
        evaluation_sha256 = sha256_file(args.evaluation_output)
    report = {
        "seed": args.seed,
        "requested_size": args.size,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "merged_unique_terms": len(entries),
        "target_records": len(targets),
        "target_spellings": len(target_terms),
        "evaluation_manifest_records": len(manifest),
        "negative_output": str(Path(args.negative_output)),
        "negative_records": negative_count,
        "negative_sha256": negative_sha256,
        "evaluation_output": str(Path(args.evaluation_output)) if args.evaluation_output else None,
        "evaluation_records": evaluation_count,
        "evaluation_sha256": evaluation_sha256,
        "sources": [asdict(item) for item in source_reports],
    }
    report_path = Path(args.report) if args.report else Path(args.negative_output).with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
