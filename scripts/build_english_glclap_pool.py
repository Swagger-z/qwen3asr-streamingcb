"""Build a deterministic English GLCLAP negative pool from train transcripts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import sha256_file, write_jsonl
from asr.contextual.glclap_english import (
    common_unigrams,
    count_english_ngrams,
    stable_english_id,
)
from asr.data.manifest import manifest_target
from asr.data.manifest_dataset import ManifestDataset


def _frequency_stratified(
    counts: dict[str, int], *, size: int, seed: int
) -> list[tuple[str, int]]:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) < size:
        raise ValueError(f"need {size} English negative terms, found {len(ranked)}")
    bucket_count = min(10, len(ranked))
    buckets = [[] for _ in range(bucket_count)]
    for rank, item in enumerate(ranked):
        buckets[min(bucket_count - 1, rank * bucket_count // len(ranked))].append(item)
    for index, bucket in enumerate(buckets):
        random.Random(f"{seed}:{index}").shuffle(bucket)
    selected: list[tuple[str, int]] = []
    offsets = [0] * bucket_count
    while len(selected) < size:
        for index, bucket in enumerate(buckets):
            if offsets[index] < len(bucket):
                selected.append(bucket[offsets[index]])
                offsets[index] += 1
                if len(selected) == size:
                    break
    return selected


def main() -> None:
    """Create a train-only English pool without reading evaluation labels."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--max-words", type=int, default=4)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--exclude-common-unigrams", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default="en-librispeech-train-neg-v1")
    args = parser.parse_args()
    datasets = [ManifestDataset(path) for path in args.manifest]
    unigram_counts, ngram_counts = count_english_ngrams(
        (
            manifest_target(record)
            for dataset in datasets
            for record in dataset
        ),
        max_words=args.max_words,
    )
    common = common_unigrams(unigram_counts, args.exclude_common_unigrams)
    eligible = {
        term: count
        for term, count in ngram_counts.items()
        if count >= args.min_count
        and any(token not in common for token in term.split())
    }
    selected = _frequency_stratified(eligible, size=args.size, seed=args.seed)
    records = [
        {
            "catalog_version": args.version,
            "id": stable_english_id("en-neg", text),
            "text": text.upper(),
            "aliases": [text] if text.upper() != text else [],
            "language": "en",
            "weight": 1.0,
            "metadata": {
                "source": "librispeech_train_transcripts",
                "frequency": count,
                "role": "training_negative",
                "used_for_training": True,
            },
        }
        for text, count in selected
    ]
    write_jsonl(args.output, records)
    report = {
        "format_version": 1,
        "purpose": "english_training_negative_pool",
        "uses_evaluation_annotations": False,
        "seed": args.seed,
        "requested_size": args.size,
        "records": len(records),
        "eligible_terms": len(eligible),
        "unique_unigrams": len(unigram_counts),
        "excluded_common_unigrams": len(common),
        "output": str(Path(args.output).resolve()),
        "sha256": sha256_file(args.output),
        "inputs": [
            dataset.protocol_summary() for dataset in datasets
        ],
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
