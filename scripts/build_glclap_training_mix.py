"""Normalize and merge corpus JSONL files into one multilingual training manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import sha256_file
from asr.data.manifest import (
    manifest_key,
    manifest_source,
    manifest_target,
)
from asr.data.manifest_dataset import ManifestDataset


def _input_spec(value: str) -> tuple[str, str, str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected CORPUS:LANGUAGE:SPLIT=PATH")
    descriptor, raw_path = value.split("=", 1)
    fields = [field.strip() for field in descriptor.split(":")]
    if len(fields) == 1:
        fields.extend(("zh", "train"))
    if len(fields) != 3 or not all(fields) or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected CORPUS:LANGUAGE:SPLIT=PATH")
    corpus, language, split = fields
    language = language.casefold()
    if language not in {"zh", "en"}:
        raise argparse.ArgumentTypeError("LANGUAGE must be zh or en")
    return corpus.casefold(), language, split, Path(raw_path.strip()).resolve()


def main() -> None:
    """Write globally namespaced records and a complete provenance report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=_input_spec)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--check-audio", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output in {path for _corpus, _language, _split, path in args.input}:
        raise ValueError("training mix output must differ from every input")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    seen_keys: set[str] = set()
    seen_sources: set[str] = set()
    input_reports = []
    corpus_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    total = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as writer:
            for corpus, language, split, path in args.input:
                dataset = ManifestDataset(path)
                input_count = 0
                for record in dataset:
                    declared_language = record.get("language")
                    declared_corpus = record.get("corpus")
                    if declared_language and str(declared_language).casefold() != language:
                        raise ValueError(f"{path}: declared language conflicts with input spec")
                    if declared_corpus and str(declared_corpus).casefold() != corpus:
                        raise ValueError(f"{path}: declared corpus conflicts with input spec")
                    old_key = manifest_key(record)
                    inferred_source_key = (
                        old_key.split(":", 1)[1]
                        if old_key.startswith(f"{corpus}:")
                        else old_key
                    )
                    source_key = str(
                        record.get("source_key", inferred_source_key)
                    ).strip()
                    key = f"{corpus}:{source_key}"
                    source = manifest_source(record)
                    target = manifest_target(record).strip()
                    if not target:
                        raise ValueError(f"{path}: empty transcript for {old_key!r}")
                    source_identity = str(Path(source).resolve()).casefold()
                    if args.check_audio and not Path(source).is_file():
                        raise ValueError(f"{path}: missing audio for {old_key!r}: {source}")
                    if key in seen_keys:
                        raise ValueError(f"duplicate namespaced key: {key!r}")
                    if source_identity in seen_sources:
                        raise ValueError(f"duplicate audio source in training mix: {source}")
                    seen_keys.add(key)
                    seen_sources.add(source_identity)
                    output_record = {
                        **record,
                        "key": key,
                        "source_key": source_key,
                        "source": source,
                        "target": target,
                        "language": language,
                        "corpus": corpus,
                        "split": split,
                    }
                    writer.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                    input_count += 1
                    total += 1
                    corpus_counts[corpus] += 1
                    language_counts[language] += 1
                input_reports.append(
                    {
                        "corpus": corpus,
                        "language": language,
                        "split": split,
                        "path": str(path),
                        "sha256": dataset.sha256,
                        "records": input_count,
                    }
                )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    if not total:
        raise ValueError("training mix is empty")
    report = {
        "format_version": 1,
        "purpose": "multilingual_glclap_training_mix",
        "records": total,
        "corpus_counts": dict(sorted(corpus_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "check_audio": args.check_audio,
        "inputs": input_reports,
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
