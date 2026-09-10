"""Run standalone offline or accumulated-audio GLCLAP Top-K retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.audio_io import read_wav_mono_float
from asr.config import load_config
from asr.contextual.glclap import AccumulatedAudioRetrievalSession, HotwordEmbeddingIndex
from asr.data.manifest import manifest_key, manifest_source, manifest_target
from asr.contextual.replay import ReplayClock
from asr.data.timed_entities import sha256_file, validate_timed_records


def parse_args() -> argparse.Namespace:
    """Parse streaming retrieval arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-dir")
    parser.add_argument("--mode", choices=("offline", "streaming"), default="streaming")
    parser.add_argument("--require-index-metadata", action="store_true",
                        help="Require INDEX.json and verify checkpoint SHA256")
    parser.add_argument("--expected-catalog", help="Also verify the catalog SHA256 in INDEX.json")
    parser.add_argument("--require-target-coverage", action="store_true",
                        help="Require unique utterance keys and gold IDs covered by the index")
    parser.add_argument("--verify-offline", action="store_true")
    parser.add_argument("--replay-mode", choices=("fast", "realtime"), default="fast")
    parser.add_argument("--require-entity-timestamps", action="store_true")
    parser.add_argument("--warmup-refreshes", type=int, default=3)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _same_hits(left: object, right: object) -> bool:
    return left == right


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_metadata(index_path: str, checkpoint_path: str, required: bool) -> dict:
    sidecar = Path(index_path + ".json")
    if not sidecar.is_file():
        if required:
            raise FileNotFoundError(f"missing index metadata: {sidecar}")
        return {}
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if required and metadata.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValueError("index/checkpoint SHA256 mismatch; rebuild the index")
    return metadata


def _validate_targets(records: list[dict], index_path: str) -> None:
    import numpy as np

    if not records:
        raise ValueError("formal evaluation manifest is empty")
    with np.load(index_path, allow_pickle=False) as payload:
        catalog_ids = set(payload["hotword_ids"].tolist())
    seen = set()
    for record in records:
        key = manifest_key(record)
        if key in seen:
            raise ValueError(f"duplicate utterance key in formal evaluation: {key}")
        seen.add(key)
        gold = record.get("target_hotword_ids", record.get("hotword_ids", []))
        if isinstance(gold, str):
            gold = [gold]
        if not gold:
            raise ValueError(f"{key}: missing nonempty target_hotword_ids")
        missing = set(gold) - catalog_ids
        if missing:
            raise ValueError(f"{key}: target IDs absent from index: {sorted(missing)}")


def _retrieve_waveform(runtime, index, waveform, *, mode, chunk_size_sec, feed_samples, top_k,
                       replay_mode="fast", replay_clock=None):
    """Return refresh batches; offline encodes one complete [N] waveform once."""

    if len(waveform) == 0:
        raise ValueError("cannot evaluate an empty waveform")
    clock = replay_clock or ReplayClock(len(waveform), feed_samples, mode=replay_mode)
    if mode == "offline":
        clock.wait_for_feed(len(waveform))
        stamp = clock.start(len(waveform))
        started = time.perf_counter()
        encoded = runtime.encode_pcm(waveform)
        encode_ms = (time.perf_counter() - started) * 1000.0
        found = index.search(encoded.frames, top_k=top_k)
        timeline = clock.finish(stamp)
        timings = {**encoded.timings_ms, **found.timings_ms, "encode_ms": encode_ms}
        timings["total_ms"] = encode_ms + float(found.timings_ms.get("search_ms", 0.0))
        timings["processing_ms"] = float(timeline["processing_sec"]) * 1000
        return [replace(found, accumulated_audio_sec=len(waveform) / 16000,
                        timings_ms=timings, chunk_id=0, is_final=True, timeline=timeline)]
    session = AccumulatedAudioRetrievalSession(
        runtime, index, chunk_size_sec=chunk_size_sec, sample_rate=16000, top_k=top_k,
        refresh_clock=clock,
    )
    batches = []
    for begin in range(0, len(waveform), feed_samples):
        clock.wait_for_feed(min(len(waveform), begin + feed_samples))
        batches.extend(session.step(waveform[begin:begin + feed_samples]))
    final = session.finish()
    if batches and batches[-1].chunk_id == final.chunk_id:
        batches[-1] = final
    else:
        batches.append(final)
    return batches


def main() -> None:
    """Write one retrieval record and optional per-refresh trace per utterance."""

    import torch
    import importlib.metadata
    from asr.contextual.glclap_runtime import build_glclap_runtime, jsonl_records

    args = parse_args()
    config = load_config(args.config, args.override)
    records = jsonl_records(args.manifest)
    stream_cfg = dict(config.get("streaming", {}))
    chunk_size_sec = float(stream_cfg.get("chunk_size_sec", 2.0))
    feed_step_ms = int(stream_cfg.get("feed_step_ms", 100))
    top_k = int(stream_cfg.get("top_k", 50))
    if feed_step_ms <= 0 or chunk_size_sec <= 0 or top_k <= 0:
        raise ValueError("feed_step_ms, chunk_size_sec and top_k must be positive")
    if args.warmup_refreshes < 0:
        raise ValueError("warmup-refreshes must be nonnegative")
    timed_records = records if args.require_entity_timestamps else [
        record for record in records if "timing_schema_version" in record
    ]
    if args.require_entity_timestamps or timed_records:
        validate_timed_records(
            [{**r, "retrieval_mode": args.mode, "chunk_size_sec": chunk_size_sec} for r in timed_records],
            check_audio=True,
        )
    output = Path(args.output)
    protected = {Path(path).resolve() for path in (args.manifest, args.checkpoint, args.index,
                                                  args.config, args.expected_catalog) if path}
    if output.resolve() in protected or Path(str(output) + ".run.json").resolve() in protected:
        raise ValueError("retrieval outputs must not overwrite input data/model files")
    index_metadata = _index_metadata(args.index, args.checkpoint, args.require_index_metadata)
    if args.expected_catalog and index_metadata.get("catalog_sha256") != _sha256(args.expected_catalog):
        raise ValueError("index/catalog SHA256 mismatch; rebuild the index")
    if args.require_target_coverage:
        _validate_targets(records, args.index)
    model, runtime, _processor, _payload = build_glclap_runtime(config, checkpoint=args.checkpoint)
    model.eval()
    index = HotwordEmbeddingIndex.load(args.index)
    index.to(next(model.adapters.parameters()).device)
    feed_samples = max(1, round(16000 * feed_step_ms / 1000))

    output.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    versions = {}
    for package in ("qwen-asr", "transformers", "torch", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    provenance = {
        "mode": args.mode, "replay_mode": args.replay_mode,
        "resolved_config": config, "versions": versions,
        "warmup_refreshes": args.warmup_refreshes,
        "clock_scope": "preloaded PCM, one serial retrieval worker, per-utterance origin",
        "latency_excludes": ["model/index loading", "WAV file reading", "warmup",
                             "result file writing", "offline verification", "input hashing/validation"],
        "implementation_sha256": {
            "decode_streaming_retrieval.py": _sha256(__file__),
            "replay.py": _sha256(Path(__file__).resolve().parents[1] / "asr/contextual/replay.py"),
            "glclap.py": _sha256(Path(__file__).resolve().parents[1] / "asr/contextual/glclap.py"),
        },
        "device": str(next(model.adapters.parameters()).device),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "checkpoint_training_config": (_payload or {}).get("config"),
        "index_metadata": index_metadata, "verify_offline": args.verify_offline,
        "inputs": {name: {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
                   for name, path in (("checkpoint", args.checkpoint), ("index", args.index),
                                      ("manifest", args.manifest))},
    }
    if args.expected_catalog:
        provenance["inputs"]["catalog"] = {
            "path": str(Path(args.expected_catalog).resolve()), "sha256": _sha256(args.expected_catalog)
        }
    Path(str(output) + ".run.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if records and args.warmup_refreshes:
        warm_audio = read_wav_mono_float(manifest_source(records[0]), 16000)
        warm_audio = warm_audio[:max(1, round(chunk_size_sec * 16000))]
        for _ in range(args.warmup_refreshes):
            warm_features = runtime.encode_pcm(warm_audio)
            index.search(warm_features.frames, top_k=top_k)
        del warm_features, warm_audio

    with output.open("w", encoding="utf-8") as writer:
        for record in records:
            utt_id = manifest_key(record)
            waveform = read_wav_mono_float(manifest_source(record), 16000)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            batches = _retrieve_waveform(
                runtime, index, waveform, mode=args.mode, chunk_size_sec=chunk_size_sec,
                feed_samples=feed_samples, top_k=top_k,
                replay_mode=args.replay_mode,
            )
            final = batches[-1]
            elapsed = time.perf_counter() - started

            peak_memory = None
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

            offline_exact = None
            if args.verify_offline and args.mode == "streaming":
                encoded = runtime.encode_pcm(waveform)
                offline = index.search(encoded.frames, top_k=top_k)
                offline_exact = _same_hits(final.hits, offline.hits)
                del encoded, offline
            serialized_batches = [batch.to_dict() for batch in batches]
            output_record = {
                **{name: record[name] for name in (
                    "timing_schema_version", "entities", "source_utt_id", "focus_mention_id",
                    "aligned_mention_id", "aligned_hotword_id", "all_target_hotword_ids",
                    "original_audio", "original_audio_sha256", "original_duration_sec",
                    "hotword_start_sec", "hotword_end_sec", "word_start_sec", "word_end_sec",
                    "leading_silence_sec", "boundary_chunk_sec", "metadata",
                    "language", "corpus", "split", "source_key",
                ) if name in record},
                "result_schema_version": 2,
                "retrieval_mode": args.mode,
                "replay_mode": args.replay_mode,
                "retrieval_top_k": top_k,
                "chunk_size_sec": chunk_size_sec,
                "feed_step_ms": feed_step_ms,
                "key": utt_id,
                "utt_id": utt_id,
                "source": manifest_source(record),
                "audio": manifest_source(record),
                "audio_sha256": (record["audio_sha256"] if record.get("timing_schema_version") == 2
                                  else sha256_file(manifest_source(record))),
                "text": manifest_target(record, required=False),
                "target_hotword_ids": record.get(
                    "target_hotword_ids", record.get("hotword_ids", [])
                ),
                "boundary_group": record.get("boundary_group", "unspecified"),
                "duration_sec": len(waveform) / 16000,
                "elapsed_sec": elapsed,
                "replay_elapsed_sec": elapsed,
                "rtf": elapsed / max(1e-9, len(waveform) / 16000),
                "rtf_scope": "replay_elapsed_over_audio_duration_includes_pacing",
                "processing_rtf": sum(float(b.timeline.get("processing_sec", 0)) for b in batches)
                                  / max(1e-9, len(waveform) / 16000),
                "peak_memory_mb": peak_memory,
                "batches": serialized_batches,
                "final_batch": final.to_dict(),
            }
            if offline_exact is not None:
                output_record["offline_final_exact_match"] = offline_exact
            writer.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            writer.flush()
            if trace_dir is not None:
                import re
                safe_key = re.sub(r"[^0-9A-Za-z_.-]+", "_", utt_id)[:100]
                suffix = hashlib.sha256(utt_id.encode()).hexdigest()[:12]
                trace_path = trace_dir / f"{safe_key}.{suffix}.jsonl"
                with trace_path.open("w", encoding="utf-8") as trace:
                    for batch in serialized_batches:
                        trace.write(json.dumps(batch, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
