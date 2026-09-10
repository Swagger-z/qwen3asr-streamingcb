"""Top-K discovery and latency metrics on validated, timestamped replay records."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from asr.data.manifest import manifest_key, manifest_source
from asr.data.forced_alignment import record_target_ids
from asr.data.timed_entities import ordered_entities, validate_timed_records

DEFAULT_KS = (1, 5, 10, 20, 50)
DEFAULT_DEADLINES_MS = (0, 100, 200, 500, 1000, 2000)
_EPS = 1e-7


def attach_entity_timing(records: Sequence[Mapping[str, Any]],
                         timed: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Attach audited annotations only to matching audio hashes, IDs and durations.

    Missing fingerprints or ambiguous old IDs are rejected. No audio paths or
    measured timelines are rewritten to make incompatible legacy traces pass.
    """
    validate_timed_records(timed)
    by_key = {manifest_key(row): row for row in timed}
    result, seen = [], set()
    for record in records:
        key = manifest_key(record)
        if key in seen or key not in by_key:
            raise ValueError(f"missing or ambiguous timing join for {key}")
        seen.add(key)
        reference = by_key[key]
        if not record.get("audio_sha256") or record["audio_sha256"] != reference["audio_sha256"]:
            raise ValueError(f"{key}: cannot verify historical audio identity; rerun retrieval")
        if Path(manifest_source(record)).resolve() != Path(manifest_source(reference)).resolve():
            raise ValueError(f"{key}: audio path mismatch; do not relabel boundary results")
        if abs(float(record.get("duration_sec", -1)) - reference["duration_sec"]) > _EPS:
            raise ValueError(f"{key}: audio duration mismatch")
        if set(record_target_ids(record)) != set(record_target_ids(reference)):
            raise ValueError(f"{key}: historical target IDs differ from timestamp annotations")
        for field in ("source_utt_id", "focus_mention_id", "boundary_group"):
            if field in record and record[field] != reference.get(field, "unspecified" if field == "boundary_group" else None):
                raise ValueError(f"{key}: historical {field} differs; cannot relabel results")
        result.append({**reference, **record, **{
            name: reference[name] for name in (
                "timing_schema_version", "entities", "source_utt_id", "focus_mention_id",
                "aligned_mention_id", "aligned_hotword_id", "boundary_group",
                "boundary_chunk_sec", "leading_silence_sec", "original_duration_sec",
                "hotword_start_sec", "hotword_end_sec", "word_start_sec", "word_end_sec",
            ) if name in reference
        }})
    if seen != set(by_key):
        raise ValueError("timing manifest and result key sets differ")
    return result


def _validate_batches(record: Mapping[str, Any], maximum_k: int) -> list[Mapping[str, Any]]:
    key = manifest_key(record)
    batches = list(record.get("batches", ()))
    if not batches or int(record.get("retrieval_top_k", 0)) < maximum_k:
        raise ValueError(f"{key}: missing batches or stored Top-K smaller than requested K")
    duration = float(record["duration_sec"])
    feed = int(round(float(record.get("feed_step_ms", 0)) * 16))
    chunk = int(round(float(record.get("chunk_size_sec", 0)) * 16000))
    total = int(round(duration * 16000))
    if feed <= 0 or chunk <= 0:
        raise ValueError(f"{key}: missing positive feed/chunk configuration")
    if record.get("retrieval_mode") == "offline":
        expected = [total]
    elif record.get("retrieval_mode") == "streaming":
        expected = list(range(chunk, total + 1, chunk))
        if not expected or expected[-1] != total:
            expected.append(total)
    else:
        raise ValueError(f"{key}: missing retrieval_mode")
    if len(expected) != len(batches):
        raise ValueError(f"{key}: refreshes missing, duplicated or dropped")
    expected_clock = {"fast": "simulated_fifo", "realtime": "monotonic_realtime"}.get(record.get("replay_mode"))
    if not expected_clock:
        raise ValueError(f"{key}: missing replay_mode; cannot infer observed latency")
    previous_finish = 0.0
    for index, (samples, batch) in enumerate(zip(expected, batches)):
        if int(batch.get("chunk_id", -1)) != index or bool(batch.get("is_final")) != (index == len(batches) - 1):
            raise ValueError(f"{key}: invalid chunk identity/final flag")
        timeline = batch.get("timeline", {})
        if timeline.get("clock") != expected_clock:
            raise ValueError(f"{key}: missing or mixed measured/simulated timeline")
        names = ("audio_cutoff_sec", "ready_sec", "start_sec", "finish_sec",
                 "processing_sec", "feed_wait_sec", "queue_wait_sec")
        try:
            t, a, s, f, processing, feed_wait, queue_wait = (float(timeline[name]) for name in names)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{key}: incomplete refresh timeline") from exc
        if not all(math.isfinite(v) for v in (t, a, s, f, processing, feed_wait, queue_wait)):
            raise ValueError(f"{key}: nonfinite timeline")
        ready = min(total, ((samples + feed - 1) // feed) * feed) / 16000
        if (abs(t - samples / 16000) > _EPS or abs(float(batch["accumulated_audio_sec"]) - t) > _EPS
                or abs(a - ready) > _EPS or a + _EPS < t or s + _EPS < max(a, previous_finish)
                or f + _EPS < s or processing < 0
                or abs(f - s - processing) > _EPS
                or abs(a - t - feed_wait) > _EPS or abs(s - a - queue_wait) > _EPS):
            raise ValueError(f"{key}: inconsistent replay clock or latency decomposition")
        if expected_clock == "simulated_fifo" and abs(s - max(a, previous_finish)) > _EPS:
            raise ValueError(f"{key}: simulated FIFO start mismatch")
        ids = [str(hit["hotword_id"]) for hit in batch.get("hits", ())]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{key}: duplicate hotword IDs in Top-K")
        if any(not math.isfinite(float(hit["score"])) for hit in batch.get("hits", ()) if "score" in hit):
            raise ValueError(f"{key}: nonfinite retrieval scores")
        if len(ids) > int(record["retrieval_top_k"]) or any(
            int(hit.get("rank", rank)) != rank
            for rank, hit in enumerate(batch.get("hits", ()), 1)
        ):
            raise ValueError(f"{key}: Top-K ranking order or length is inconsistent")
        previous_finish = f
    if record.get("final_batch") is not None and record["final_batch"] != batches[-1]:
        raise ValueError(f"{key}: final_batch differs from the last refresh")
    return batches


def _rate(values: Sequence[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def _summary(rows: Sequence[dict], deadlines: Sequence[int]) -> dict:
    primary = [row for row in rows if row["primary"]]
    repeated = [row for row in rows if not row["primary"]]
    found = [row for row in primary if row["detected"]]
    observed = [row for row in found if row["dropout"] is not None]
    output = {
        "target_count": len(primary), "detected_count": len(found),
        "missed_count": len(primary) - len(found),
        "miss_rate": _rate([not row["detected"] for row in primary]),
        "first_complete_refresh_recall": _rate([row["first_complete_hit"] for row in primary]),
        "deadline_recall": {str(d): _rate([row["deadline_hits"][str(d)] for row in primary]) for d in deadlines},
        "early_before_onset_rate": _rate([row["early_before_onset"] for row in primary]),
        "early_during_word_rate": _rate([row["early_during_word"] for row in primary]),
        "post_detection_observed_count": len(observed),
        "dropout_rate": _rate([row["dropout"] for row in observed]),
        "repeat_mention_count": len(repeated),
        "repeat_candidate_availability": _rate([row["first_complete_hit"] for row in repeated]),
    }
    for field in ("latency_ms", "audio_wait_ms", "feed_wait_ms", "queue_wait_ms", "processing_ms"):
        values = [row[field] for row in found]
        output[field + "_mean"] = float(np.mean(values)) if values else None
        output[field + "_p50"] = float(np.percentile(values, 50)) if values else None
        output[field + "_p95"] = float(np.percentile(values, 95)) if values else None
    return output


def _cluster_ci(differences: Mapping[str, list[float]], samples: int, seed: int) -> dict:
    """Bootstrap original utterances, retaining all their paired focus entities."""
    if not differences:
        return {"estimate": None, "ci95_low": None, "ci95_high": None,
                "source_utterance_count": 0, "paired_focus_count": 0}
    ordered = [differences[key] for key in sorted(differences)]
    sums = np.asarray([sum(values) for values in ordered], dtype=np.float64)
    counts = np.asarray([len(values) for values in ordered], dtype=np.int64)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        selected = rng.integers(0, len(ordered), size=len(ordered))
        means.append(float(sums[selected].sum() / counts[selected].sum()))
    return {"estimate": float(sums.sum() / counts.sum()),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
            "source_utterance_count": len(ordered), "paired_focus_count": int(counts.sum())}


def _boundary_pairs(rows: Sequence[dict], deadlines: Sequence[int], samples: int, seed: int) -> dict:
    groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["primary"] and row["boundary_group"] in {"center", "cross-25", "cross-50", "cross-75"}:
            pair = (row["source_utt_id"], row["mention_id"])
            if row["boundary_group"] in groups[pair]:
                raise ValueError(f"duplicate boundary pair/condition: {pair}")
            groups[pair][row["boundary_group"]] = row
    paired = {pair: group for pair, group in groups.items()
              if set(group) == {"center", "cross-25", "cross-50", "cross-75"}}
    output = {"unpaired_focus_count": len(groups) - len(paired)}
    for metric in ("first_complete_refresh", *[f"deadline_{d}ms" for d in deadlines]):
        by_source: dict[str, list[float]] = defaultdict(list)
        for (source, _mention), group in paired.items():
            def value(row):
                if metric == "first_complete_refresh":
                    return float(row["first_complete_hit"])
                return float(row["deadline_hits"][metric[9:-2]])
            difference = value(group["center"]) - np.mean([value(group[name]) for name in
                                                           ("cross-25", "cross-50", "cross-75")])
            by_source[source].append(float(difference))
        output[metric] = _cluster_ci(by_source, samples, seed)
    return output


def evaluate_online_records(records: Sequence[Mapping[str, Any]], *,
                            ks: Sequence[int] = DEFAULT_KS,
                            deadlines_ms: Sequence[int] = DEFAULT_DEADLINES_MS,
                            bootstrap_samples: int = 2000, seed: int = 42) -> tuple[dict, list[dict], list[dict]]:
    """Return online summary, one row per mention/K, and per-refresh recall rows.

    Discovery uses the first occurrence of each ID in each original utterance.
    Boundary records evaluate only their focus mention. A full timed entity list
    is still required so a repeated focus cannot masquerade as first discovery.
    Misses remain in recall denominators; undefined latency summaries are null.
    """
    ks, deadlines = tuple(ks), tuple(deadlines_ms)
    if not ks or any(k <= 0 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("ks must be distinct positive integers")
    if not deadlines or any(d < 0 for d in deadlines) or len(set(deadlines)) != len(deadlines):
        raise ValueError("deadlines must be distinct nonnegative milliseconds")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    validate_timed_records(records)
    modes = {(row.get("replay_mode"), row.get("retrieval_mode")) for row in records}
    if len(modes) != 1:
        raise ValueError("evaluate each retrieval/replay mode separately")
    entity_rows, refresh_rows = [], []
    for record in records:
        key = manifest_key(record)
        batches = _validate_batches(record, max(ks))
        entities = ordered_entities(record)
        focus = record.get("focus_mention_id")
        if focus is not None:
            entities = [entity for entity in entities if entity["mention_id"] == focus]
        for batch in batches:
            t = float(batch["timeline"]["audio_cutoff_sec"])
            eligible = [entity for entity in entities if entity["is_first_occurrence"] and entity["end_sec"] <= t + _EPS]
            row = {"key": key, "source_utt_id": record.get("source_utt_id", key),
                   "language": record.get("language", "unknown"),
                   "corpus": record.get("corpus", record.get("dataset", "unknown")),
                   "chunk_id": batch["chunk_id"], "boundary_group": record.get("boundary_group", "unspecified"),
                   "completed_primary_count": len(eligible), **batch["timeline"],
                   "is_final": batch["is_final"], "frame_count": batch.get("frame_count"),
                   "timings_ms": dict(batch.get("timings_ms", {}))}
            for k in ks:
                predicted = {str(hit["hotword_id"]) for hit in batch["hits"][:k]}
                row[f"recall_at_{k}"] = _rate([entity["hotword_id"] in predicted for entity in eligible])
            refresh_rows.append(row)
        for entity in entities:
            start, end = float(entity["start_sec"]), float(entity["end_sec"])
            eligible = [j for j, batch in enumerate(batches) if batch["timeline"]["audio_cutoff_sec"] + _EPS >= end]
            for k in ks:
                hits = [any(str(hit["hotword_id"]) == entity["hotword_id"] for hit in batch["hits"][:k])
                        for batch in batches]
                first = next((j for j in eligible if hits[j]), None)
                primary = entity["is_first_occurrence"]
                row = {"key": key, "source_utt_id": record.get("source_utt_id", key),
                       "language": record.get("language", "unknown"),
                       "corpus": record.get("corpus", record.get("dataset", "unknown")),
                       "mention_id": entity["mention_id"], "hotword_id": entity["hotword_id"],
                       "start_sec": start, "end_sec": end, "k": k, "primary": primary,
                       "boundary_group": record.get("boundary_group", "unspecified"),
                       "first_complete_hit": bool(eligible and hits[eligible[0]]),
                       "first_complete_chunk_id": batches[eligible[0]]["chunk_id"] if eligible else None,
                       "early_before_onset": any(hit and batch["timeline"]["audio_cutoff_sec"] < start
                                                for hit, batch in zip(hits, batches)),
                       "early_during_word": any(hit and start <= batch["timeline"]["audio_cutoff_sec"] < end
                                               for hit, batch in zip(hits, batches)),
                       "detected": first is not None if primary else None,
                       "first_detected_chunk_id": None,
                       "latency_ms": None, "audio_wait_ms": None, "feed_wait_ms": None,
                       "queue_wait_ms": None, "processing_ms": None, "dropout": None,
                       "deadline_hits": {str(d): False if primary else None for d in deadlines}}
                if primary and first is not None:
                    stamp = batches[first]["timeline"]
                    row.update({
                        "first_detected_chunk_id": batches[first]["chunk_id"],
                        "latency_ms": max(0.0, stamp["finish_sec"] - end) * 1000,
                        "audio_wait_ms": max(0.0, stamp["audio_cutoff_sec"] - end) * 1000,
                        "feed_wait_ms": stamp["feed_wait_sec"] * 1000,
                        "queue_wait_ms": stamp["queue_wait_sec"] * 1000,
                        "processing_ms": stamp["processing_sec"] * 1000,
                        "dropout": any(not hit for hit in hits[first + 1:]) if first + 1 < len(hits) else None,
                        "deadline_hits": {str(d): stamp["finish_sec"] - end <= d / 1000 + _EPS for d in deadlines},
                    })
                entity_rows.append(row)
    summary = {"schema_version": 1, "replay_mode": next(iter(modes))[0],
               "retrieval_mode": next(iter(modes))[1], "ks": list(ks),
               "deadlines_ms": list(deadlines), "utterances": len(records),
               "bootstrap_unit": "source_utt_id", "bootstrap_samples": bootstrap_samples,
               "seed": seed, "by_k": {}, "by_boundary": {}, "boundary_penalty": {}}
    services = [float(row["processing_sec"]) for row in refresh_rows]
    summary["compute"] = {
        "refresh_count": len(services),
        "processing_total_sec": sum(services),
        "input_audio_total_sec": sum(float(r["duration_sec"]) for r in records),
        "encoded_audio_total_sec": sum(float(r["audio_cutoff_sec"]) for r in refresh_rows),
        "processing_ms_mean": float(np.mean(services) * 1000),
        "processing_ms_p50": float(np.percentile(services, 50) * 1000),
        "processing_ms_p95": float(np.percentile(services, 95) * 1000),
    }
    summary["compute"]["processing_rtf"] = (
        sum(services) / summary["compute"]["input_audio_total_sec"]
    )
    for k in ks:
        rows = [row for row in entity_rows if row["k"] == k]
        summary["by_k"][str(k)] = _summary(rows, deadlines)
        summary["by_boundary"][str(k)] = {
            label: _summary([row for row in rows if row["boundary_group"] == label], deadlines)
            for label in sorted({row["boundary_group"] for row in rows})
        }
        summary["boundary_penalty"][str(k)] = _boundary_pairs(rows, deadlines, bootstrap_samples, seed)
    return summary, entity_rows, refresh_rows
