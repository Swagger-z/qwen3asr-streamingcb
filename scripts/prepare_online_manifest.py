"""Validate/upgrade offline alignments into grouped and focused timed manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import write_jsonl
from asr.data.timed_entities import prepare_timed_records, sha256_file, validate_timed_records


def read_records(path: str) -> list[dict]:
    """Read JSONL, resolving each present audio alias against its manifest."""
    source = Path(path).resolve()
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        for field in ("source", "audio"):
            if field in row:
                audio = Path(row[field]).expanduser()
                row[field] = str((audio if audio.is_absolute() else source.parent / audio).resolve())
    return rows


def main() -> None:
    """Prepare audited timestamps without running or training any ASR model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--aligned-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aligned-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    inputs = {Path(args.source_manifest).resolve(), Path(args.aligned_manifest).resolve()}
    outputs = [Path(path).resolve() for path in (args.output, args.aligned_output, args.report)]
    if len(set(outputs)) != 3 or inputs & set(outputs):
        raise ValueError("input and output paths must be distinct")
    for path in outputs:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {path}; use --overwrite")
    grouped, focused = prepare_timed_records(read_records(args.source_manifest),
                                              read_records(args.aligned_manifest))
    validate_timed_records(grouped)
    validate_timed_records(focused)
    write_jsonl(args.output, grouped)
    write_jsonl(args.aligned_output, focused)
    report = {"timing_schema_version": 2, "utterances": len(grouped), "mentions": len(focused),
              "source_manifest_sha256": sha256_file(args.source_manifest),
              "aligned_manifest_sha256": sha256_file(args.aligned_manifest),
              "output_sha256": sha256_file(args.output),
              "aligned_output_sha256": sha256_file(args.aligned_output)}
    outputs[2].parent.mkdir(parents=True, exist_ok=True)
    outputs[2].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
