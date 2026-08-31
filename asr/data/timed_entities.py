"""Validated entity timestamps and stable identities for online retrieval."""

from __future__ import annotations

import hashlib
import math
import re
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import manifest_key, manifest_source
from .forced_alignment import record_target_ids, record_target_mentions

TIMING_SCHEMA_VERSION = 2


def stable_record_key(source_id: str, mention_id: str, condition: str = "") -> str:
    """Create a filesystem-safe, collision-resistant identity for a variant."""
    stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", source_id)[:80] or "utterance"
    digest = hashlib.sha256(f"{source_id}\0{mention_id}".encode()).hexdigest()[:16]
    return f"{stem}__mention_{digest}" + (f"__{condition}" if condition else "")


def sha256_file(path: str | Path) -> str:
    """Hash an input file without materializing it in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_fingerprint(path: str | Path) -> dict[str, Any]:
    """Return PCM WAV duration and content identity, outside inference timing."""
    with wave.open(str(path), "rb") as reader:
        if reader.getcomptype() != "NONE" or reader.getsampwidth() != 2:
            raise ValueError("timed evaluation requires uncompressed PCM16 WAV")
        if reader.getframerate() != 16000:
            raise ValueError("timed evaluation requires 16 kHz WAV")
        duration = reader.getnframes() / reader.getframerate()
    return {"duration_sec": duration, "audio_sha256": sha256_file(path)}


def ordered_entities(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return timed mentions with first-occurrence flags computed from all mentions."""
    entities = [dict(item) for item in record.get("entities", ())]
    entities.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"]),
                                    str(item["mention_id"])))
    seen: set[str] = set()
    for item in entities:
        item["is_first_occurrence"] = item["hotword_id"] not in seen
        seen.add(item["hotword_id"])
    return entities


def prepare_timed_records(sources: Sequence[Mapping[str, Any]],
                          aligned: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Join original gold mentions to alignments; emit grouped and focus views.

    Original gold determines completeness. This also upgrades old aligner files
    whose canonical key was not changed when a sentence expanded into mentions.
    It does not repair decoded results or trust a mismatching audio alias.
    """
    by_mention: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in aligned:
        pair = (str(row.get("source_utt_id", row.get("utt_id", ""))),
                str(row.get("aligned_mention_id", row.get("aligned_hotword_id", ""))))
        if not all(pair) or pair in by_mention:
            raise ValueError(f"missing or duplicate alignment identity: {pair}")
        by_mention[pair] = row
    grouped, focused = [], []
    used, source_keys = set(), set()
    fingerprints: dict[str, dict] = {}
    for original in sources:
        source_id = manifest_key(original)
        if source_id in source_keys:
            raise ValueError(f"duplicate source key: {source_id}")
        source_keys.add(source_id)
        audio = str(Path(manifest_source(original)).resolve())
        if "audio" in original and str(Path(str(original["audio"])).resolve()) != audio:
            raise ValueError(f"{source_id}: conflicting source/audio paths")
        entities = []
        for mention in record_target_mentions(original):
            pair = (source_id, str(mention["mention_id"]))
            if pair not in by_mention:
                raise ValueError(f"missing alignment: {pair}")
            row = by_mention[pair]
            used.add(pair)
            for field in ("source", "audio"):
                if field in row and str(Path(str(row[field])).resolve()) != audio:
                    raise ValueError(f"{pair}: alignment audio differs from source manifest")
            if str(row.get("aligned_hotword_id")) != str(mention["hotword_id"]):
                raise ValueError(f"{pair}: alignment hotword ID mismatch")
            entities.append({**mention, "start_sec": float(row["hotword_start_sec"]),
                             "end_sec": float(row["hotword_end_sec"])})
        if audio not in fingerprints:
            fingerprints[audio] = audio_fingerprint(audio)
        row = {**original, "key": source_id, "utt_id": source_id, "source_utt_id": source_id,
               "source": audio, "audio": audio, "original_audio": audio,
               "timing_schema_version": TIMING_SCHEMA_VERSION,
               "entities": entities, **fingerprints[audio]}
        row["entities"] = ordered_entities(row)
        validate_timed_record(row)
        grouped.append(row)
        for entity in row["entities"]:
            key = stable_record_key(source_id, entity["mention_id"])
            focus = {**row, "key": key, "utt_id": key,
                     "target_hotword_ids": [entity["hotword_id"]],
                     "all_target_hotword_ids": list(record_target_ids(original)),
                     "focus_mention_id": entity["mention_id"],
                     "aligned_mention_id": entity["mention_id"],
                     "aligned_hotword_id": entity["hotword_id"],
                     "hotword_start_sec": entity["start_sec"],
                     "hotword_end_sec": entity["end_sec"]}
            focused.append(focus)
    if used != set(by_mention):
        raise ValueError(f"unexpected alignments: {sorted(set(by_mention) - used)[:5]}")
    if not grouped:
        raise ValueError("empty timed manifest")
    return grouped, focused


def validate_boundary(record: Mapping[str, Any]) -> None:
    """Check actual shifted focus placement, including center containment."""
    label = str(record.get("boundary_group", ""))
    if label in {"", "unspecified"}:
        return
    chunk = float(record.get("boundary_chunk_sec", 0))
    silence = float(record.get("leading_silence_sec", -1))
    start, end = float(record["hotword_start_sec"]), float(record["hotword_end_sec"])
    if not math.isfinite(chunk) or chunk <= 0 or not math.isfinite(silence) or silence < 0:
        raise ValueError("missing or invalid boundary chunk/silence metadata")
    if record.get("retrieval_mode") == "streaming" and abs(float(record["chunk_size_sec"]) - chunk) > 1e-7:
        raise ValueError("retrieval chunk differs from boundary generation; regenerate variants")
    tolerance = 2 / 16000
    if label == "center":
        middle = (start + end) / 2
        cell = round((middle - chunk / 2) / chunk)
        valid = (abs(middle - (cell + 0.5) * chunk) <= tolerance
                 and start >= cell * chunk - tolerance
                 and end <= (cell + 1) * chunk + tolerance)
    elif label in {"b-400", "b-200", "b-100"}:
        offset = int(label[2:]) / 1000
        if offset >= chunk:
            raise ValueError("before-boundary offset must be smaller than chunk duration")
        landmark = (start + end) / 2 + offset
        valid = abs(landmark - round(landmark / chunk) * chunk) <= tolerance
    elif label in {"cross-25", "cross-50", "cross-75"}:
        after = int(label[6:]) / 100
        landmark = start + (end - start) * (1 - after)
        boundary = round(landmark / chunk) * chunk
        valid = start < boundary < end and abs(landmark - boundary) <= tolerance
    else:
        raise ValueError(f"unknown boundary condition: {label}")
    if not valid:
        raise ValueError(f"focus span does not satisfy boundary condition {label}")


def validate_timed_record(record: Mapping[str, Any], *, check_audio: bool = False) -> None:
    """Reject missing, conflicting or out-of-range times instead of inventing zero."""
    key = manifest_key(record)
    if record.get("timing_schema_version") != TIMING_SCHEMA_VERSION:
        raise ValueError(f"{key}: require timing_schema_version=2; prepare timed manifests first")
    if record.get("utt_id", key) != key:
        raise ValueError(f"{key}: key/utt_id mismatch")
    audio = str(Path(manifest_source(record)).resolve())
    if "audio" in record and str(Path(str(record["audio"])).resolve()) != audio:
        raise ValueError(f"{key}: source/audio mismatch; regenerate boundary data")
    duration = float(record.get("duration_sec", -1))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{key}: missing positive duration_sec")
    entities, seen = list(record.get("entities", ())), set()
    if not entities:
        raise ValueError(f"{key}: missing timed entities")
    for entity in entities:
        mid, hid = str(entity.get("mention_id", "")), str(entity.get("hotword_id", ""))
        if not mid or not hid or mid in seen:
            raise ValueError(f"{key}: missing or duplicate entity identity")
        seen.add(mid)
        start, end = float(entity.get("start_sec", -1)), float(entity.get("end_sec", -1))
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 1e-7):
            raise ValueError(f"{key}/{mid}: invalid entity timestamps")
    gold = set(record_target_ids(record))
    if not gold or not gold <= {str(e["hotword_id"]) for e in entities}:
        raise ValueError(f"{key}: gold IDs lack timed entities")
    focus = record.get("focus_mention_id")
    if focus is not None:
        matches = [e for e in entities if e["mention_id"] == focus]
        if len(matches) != 1 or gold != {matches[0]["hotword_id"]}:
            raise ValueError(f"{key}: invalid focus mention")
        for field, actual in (("hotword_start_sec", matches[0]["start_sec"]),
                              ("hotword_end_sec", matches[0]["end_sec"]),
                              ("word_start_sec", matches[0]["start_sec"]),
                              ("word_end_sec", matches[0]["end_sec"])):
            if field in record:
                value = float(record[field])
                if not math.isfinite(value) or abs(value - float(actual)) > 1e-7:
                    raise ValueError(f"{key}: conflicting {field}")
    elif gold != {str(e["hotword_id"]) for e in entities}:
        raise ValueError(f"{key}: unscoped extra timed entities")
    if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("audio_sha256", ""))):
        raise ValueError(f"{key}: missing audio fingerprint")
    validate_boundary(record)
    if record.get("boundary_group", "unspecified") not in {"", "unspecified"}:
        if not record.get("source_utt_id") or focus is None:
            raise ValueError(f"{key}: boundary records require source identity and focus mention")
        expected = float(record["original_duration_sec"]) + float(record["leading_silence_sec"])
        if not math.isfinite(expected) or abs(duration - expected) > 2 / 16000:
            raise ValueError(f"{key}: shifted WAV duration does not match leading silence")
    if check_audio:
        actual = audio_fingerprint(audio)
        if actual["audio_sha256"] != record["audio_sha256"] or abs(actual["duration_sec"] - duration) > 1 / 16000:
            raise ValueError(f"{key}: audio changed since timestamp preparation")


def validate_timed_records(records: Sequence[Mapping[str, Any]], *, check_audio: bool = False) -> None:
    """Validate a nonempty manifest and unique per-record identities."""
    if not records:
        raise ValueError("empty timed manifest")
    seen = set()
    for record in records:
        key = manifest_key(record)
        if key in seen:
            raise ValueError(f"duplicate timed record key: {key}")
        seen.add(key)
        validate_timed_record(record, check_audio=check_audio)
