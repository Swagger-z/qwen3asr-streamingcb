"""Validate an aligned hotword manifest before boundary-stress generation."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog import HotwordCatalog
from asr.data.forced_alignment import record_target_ids, record_target_mentions


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return records


def _audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as reader:
        if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
            raise ValueError(f"stage2 requires uncompressed PCM16 WAV: {path}")
        return reader.getnframes() / reader.getframerate()


def main() -> None:
    """Check expected keys, catalog IDs, time spans, duplicates, and WAV duration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--aligned-manifest", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-audio-check", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source_manifest).resolve()
    aligned_path = Path(args.aligned_manifest).resolve()
    catalog = HotwordCatalog.from_jsonl(args.catalog)
    expected: set[tuple[str, str]] = set()
    for record in _jsonl(source_path):
        source_id = str(record.get("utt_id", ""))
        if not source_id:
            raise ValueError("source manifest contains a record without utt_id")
        for mention in record_target_mentions(record):
            hotword_id = str(mention["hotword_id"])
            if hotword_id not in catalog.entries:
                raise ValueError(f"{source_id}: target ID not in catalog: {hotword_id}")
            expected.add((source_id, str(mention["mention_id"])))

    actual: set[tuple[str, str]] = set()
    durations: dict[Path, float] = {}
    for line_number, record in enumerate(_jsonl(aligned_path), 1):
        source_id = str(record.get("source_utt_id", record.get("utt_id", "")))
        hotword_id = str(record.get("aligned_hotword_id", ""))
        mention_id = str(record.get("aligned_mention_id", hotword_id))
        key = (source_id, mention_id)
        if not source_id or not hotword_id or not mention_id:
            raise ValueError(f"{aligned_path}:{line_number}: missing aligned key")
        if key in actual:
            raise ValueError(f"{aligned_path}:{line_number}: duplicate aligned key {key}")
        actual.add(key)
        if hotword_id not in catalog.entries:
            raise ValueError(f"{aligned_path}:{line_number}: unknown hotword ID {hotword_id}")
        if record_target_ids(record) != (hotword_id,):
            raise ValueError(f"{aligned_path}:{line_number}: record must contain exactly its aligned ID")
        start = float(record.get("hotword_start_sec", -1))
        end = float(record.get("hotword_end_sec", -1))
        if not 0 <= start < end:
            raise ValueError(f"{aligned_path}:{line_number}: invalid hotword span {start}, {end}")
        if not args.skip_audio_check:
            audio_path = Path(str(record.get("audio", ""))).expanduser()
            if not audio_path.is_absolute():
                audio_path = aligned_path.parent / audio_path
            audio_path = audio_path.resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"{aligned_path}:{line_number}: missing audio {audio_path}")
            duration = durations.setdefault(audio_path, _audio_duration(audio_path))
            if end > duration + 0.001:
                raise ValueError(
                    f"{aligned_path}:{line_number}: hotword ends at {end}s after audio duration {duration}s"
                )

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if extra:
        raise ValueError(f"aligned manifest has unexpected source/mention keys: {extra[:10]}")
    if missing and not args.allow_incomplete:
        raise ValueError(f"aligned manifest is incomplete; missing mention keys: {missing[:10]}")
    report = {
        "format_version": 1,
        "source_manifest": str(source_path),
        "aligned_manifest": str(aligned_path),
        "expected_aligned_records": len(expected),
        "actual_aligned_records": len(actual),
        "missing_count": len(missing),
        "missing": missing,
        "audio_file_count": len(durations),
        "complete": not missing,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
