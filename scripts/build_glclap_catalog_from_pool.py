"""Combine gold targets with a same-language, transcript-safe distractor pool."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import load_jsonl, sha256_file, write_jsonl
from asr.contextual.glclap_data import batch_negative_exclusions, normalize_term_for_language
from asr.data.manifest import manifest_target


def main() -> None:
    """Build a fixed-size monolingual evaluation catalog."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-catalog", required=True)
    parser.add_argument("--distractor-pool", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--language", choices=("zh", "en"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    targets = load_jsonl(args.target_catalog)
    distractors = load_jsonl(args.distractor_pool)
    manifest = load_jsonl(args.eval_manifest)
    target_ids = {str(record["id"]) for record in targets}
    if len(target_ids) != len(targets):
        raise ValueError("target catalog contains duplicate IDs")
    manifest_ids = {
        str(value)
        for record in manifest
        for value in record.get("target_hotword_ids", [])
    }
    if not manifest_ids <= target_ids:
        raise ValueError("evaluation manifest references IDs absent from target catalog")
    if len(targets) > args.size:
        raise ValueError("target catalog is larger than requested evaluation catalog")
    target_spellings = {
        normalize_term_for_language(value, args.language)
        for record in targets
        for value in (record.get("text", ""), *record.get("aliases", []))
    }
    transcripts = [manifest_target(record) for record in manifest]
    spoken_terms = batch_negative_exclusions(
        transcripts,
        language=args.language,
        min_chars=1,
        max_chars=8,
        min_words=1,
        max_words=4,
    )
    eligible = []
    seen_terms = set(target_spellings)
    seen_ids = set(target_ids)
    for record in distractors:
        text = str(record.get("text", ""))
        normalized = normalize_term_for_language(text, args.language)
        if not normalized or normalized in seen_terms or str(record.get("id")) in seen_ids:
            continue
        if normalized in spoken_terms:
            continue
        seen_terms.add(normalized)
        seen_ids.add(str(record["id"]))
        eligible.append(record)
    needed = args.size - len(targets)
    if len(eligible) < needed:
        raise ValueError(f"need {needed} clean distractors, found {len(eligible)}")
    random.Random(args.seed).shuffle(eligible)
    records = []
    for role, source_records in (("evaluation_target", targets), ("evaluation_distractor", eligible[:needed])):
        for source in source_records:
            records.append(
                {
                    **source,
                    "catalog_version": args.version,
                    "language": args.language,
                    "metadata": {**dict(source.get("metadata", {})), "role": role},
                }
            )
    records.sort(key=lambda record: (record["metadata"]["role"] != "evaluation_target", str(record["id"])))
    write_jsonl(args.output, records)
    report = {
        "format_version": 1,
        "purpose": "monolingual_evaluation_catalog",
        "language": args.language,
        "records": len(records),
        "targets": len(targets),
        "distractors": needed,
        "output_sha256": sha256_file(args.output),
        "inputs": {
            "target_catalog_sha256": sha256_file(args.target_catalog),
            "distractor_pool_sha256": sha256_file(args.distractor_pool),
            "eval_manifest_sha256": sha256_file(args.eval_manifest),
        },
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
