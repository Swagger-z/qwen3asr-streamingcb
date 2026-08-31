"""Build audited, uniquely identified center/cross-boundary PCM16 WAVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.data.boundary_stress import build_variants
from asr.data.manifest import manifest_key, manifest_source
from asr.data.timed_entities import (
    audio_fingerprint, prepare_timed_records, stable_record_key,
    validate_timed_record, validate_timed_records,
)
from scripts.prepare_online_manifest import read_records


def main() -> None:
    """Shift all mention timestamps together and preserve focus identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--chunk-ms", type=int, default=2000)
    parser.add_argument("--source-manifest", help="Original gold manifest for upgrading old alignments")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    records = read_records(args.manifest)
    if args.source_manifest:
        _, records = prepare_timed_records(read_records(args.source_manifest), records)
    validate_timed_records(records, check_audio=True)
    output_dir, output_manifest = Path(args.output_dir), Path(args.output_manifest)
    inputs = {Path(args.manifest).resolve(), Path(args.source_manifest or args.manifest).resolve()}
    if output_manifest.resolve() in inputs:
        raise ValueError("boundary output must not overwrite input manifests")
    if not args.overwrite:
        if output_manifest.exists():
            raise FileExistsError("boundary manifest exists; use new output or --overwrite")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("boundary WAV directory is not empty; use new output or --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with output_manifest.open("w", encoding="utf-8") as target:
        for record in records:
            focus = record.get("focus_mention_id", record.get("aligned_mention_id"))
            if not focus:
                raise ValueError("boundary input must be the focused aligned manifest")
            source_id = record["source_utt_id"]
            unique = stable_record_key(source_id, focus)
            variants = build_variants(
                manifest_source(record), output_dir / unique,
                float(record["hotword_start_sec"]), float(record["hotword_end_sec"]),
                args.chunk_ms / 1000,
            )
            for variant in variants:
                shift = float(variant["leading_silence_sec"])
                key = stable_record_key(source_id, focus, variant["boundary_group"])
                if key in seen:
                    raise ValueError(f"duplicate boundary key: {key}")
                seen.add(key)
                entities = [
                    {**entity, "start_sec": float(entity["start_sec"]) + shift,
                     "end_sec": float(entity["end_sec"]) + shift}
                    for entity in record["entities"]
                ]
                shifted = {
                    **record, **variant, "key": key, "utt_id": key,
                    "focus_mention_id": focus, "entities": entities,
                    "all_entities": entities,
                    "original_audio": manifest_source(record),
                    "original_audio_sha256": record["audio_sha256"],
                    "original_duration_sec": record["duration_sec"],
                    **audio_fingerprint(variant["audio"]),
                }
                validate_timed_record(shifted)
                target.write(json.dumps(shifted, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
