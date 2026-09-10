"""Create deterministic synthetic rare-phrase labels for LibriSpeech eval JSONL."""

from __future__ import annotations

import argparse
import json
import sys
import hashlib
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import sha256_file, write_jsonl
from asr.contextual.glclap_english import (
    common_unigrams,
    normalized_english_tokens,
    select_synthetic_phrases,
    stable_english_id,
)
from asr.data.manifest import manifest_key, manifest_source, manifest_target
from asr.data.manifest_dataset import ManifestDataset
from asr.contextual.glclap_runtime import load_negative_vocabulary
from asr.contextual.glclap_data import normalize_english_term


def main() -> None:
    """Label one frozen LibriSpeech dev/test split using train-only statistics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", action="append", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--target-catalog-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=4)
    parser.add_argument("--exclude-common-unigrams", type=int, default=5000)
    parser.add_argument(
        "--target-vocabulary",
        help="optional train-derived catalog constraining the shared eval target vocabulary",
    )
    parser.add_argument("--max-target-vocabulary", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_sets = [ManifestDataset(path) for path in args.train_manifest]
    frequencies: Counter[str] = Counter()
    for dataset in train_sets:
        for record in dataset:
            frequencies.update(normalized_english_tokens(manifest_target(record)))
    common = common_unigrams(frequencies, args.exclude_common_unigrams)
    allowed_terms = None
    if args.target_vocabulary:
        vocabulary = {
            normalize_english_term(value)
            for value in load_negative_vocabulary(args.target_vocabulary)
            if normalize_english_term(value)
        }
        ranked = sorted(
            vocabulary,
            key=lambda value: hashlib.sha256(
                f"{args.seed}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        allowed_terms = set(ranked[: args.max_target_vocabulary])
    eval_set = ManifestDataset(args.eval_manifest)
    output_records: list[dict] = []
    catalog_texts: dict[str, str] = {}
    for source_record in eval_set:
        original_key = manifest_key(source_record)
        inferred_source_key = (
            original_key.split(":", 1)[1]
            if original_key.startswith("librispeech:")
            else original_key
        )
        source_key = str(
            source_record.get("source_key", inferred_source_key)
        )
        key = f"librispeech:{source_key}"
        transcript = manifest_target(source_record)
        phrases = select_synthetic_phrases(
            transcript,
            training_unigram_counts=frequencies,
            excluded_common=common,
            utt_id=key,
            max_targets=args.max_targets,
            max_words=args.max_words,
            seed=args.seed,
            allowed_terms=allowed_terms,
        )
        if not phrases:
            continue
        entities = []
        target_ids = []
        for index, phrase in enumerate(phrases):
            hotword_id = stable_english_id("en-librispeech", phrase)
            catalog_texts.setdefault(hotword_id, phrase)
            target_ids.append(hotword_id)
            entities.append(
                {
                    "id": hotword_id,
                    "hotword_id": hotword_id,
                    "mention_id": f"{key}#synthetic-{index:02d}",
                    "text": phrase,
                    "type": "synthetic_phrase",
                    "entity_type": "synthetic_phrase",
                }
            )
        output_records.append(
            {
                "key": key,
                "source_key": source_key,
                "source": manifest_source(source_record),
                "target": transcript,
                "language": "en",
                "corpus": "librispeech",
                "split": args.split,
                "entities": entities,
                "target_hotword_ids": target_ids,
                "metadata": {
                    "synthetic_hotwords": True,
                    "selection_uses_train_statistics_only": True,
                },
            }
        )
    version = f"librispeech-{args.split}-synthetic-targets-v1"
    catalog_records = [
        {
            "catalog_version": version,
            "id": hotword_id,
            "text": text,
            "aliases": [],
            "language": "en",
            "weight": 1.0,
            "metadata": {
                "source": "librispeech",
                "split": args.split,
                "role": "evaluation_target",
                "synthetic": True,
            },
        }
        for hotword_id, text in sorted(catalog_texts.items())
    ]
    if not output_records:
        raise ValueError("no LibriSpeech evaluation utterances contain eligible rare phrases")
    write_jsonl(args.manifest_output, output_records)
    write_jsonl(args.target_catalog_output, catalog_records)
    report = {
        "format_version": 1,
        "purpose": "librispeech_synthetic_hotword_evaluation",
        "split": args.split,
        "seed": args.seed,
        "eval_input": eval_set.protocol_summary(),
        "train_inputs": [dataset.protocol_summary() for dataset in train_sets],
        "input_utterances": len(eval_set),
        "evaluable_utterances": len(output_records),
        "target_terms": len(catalog_records),
        "target_mentions": sum(len(row["entities"]) for row in output_records),
        "target_vocabulary": str(Path(args.target_vocabulary).resolve()) if args.target_vocabulary else None,
        "allowed_target_terms": len(allowed_terms) if allowed_terms is not None else None,
        "manifest_sha256": sha256_file(args.manifest_output),
        "catalog_sha256": sha256_file(args.target_catalog_output),
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
