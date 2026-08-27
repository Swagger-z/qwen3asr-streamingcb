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


def _retrieve_waveform(runtime, index, waveform, *, mode, chunk_size_sec, feed_samples, top_k):
    """Return refresh batches; offline encodes one complete [N] waveform once."""

    if mode == "offline":
        started = time.perf_counter()
        encoded = runtime.encode_pcm(waveform)
        encode_ms = (time.perf_counter() - started) * 1000.0
        found = index.search(encoded.frames, top_k=top_k)
        timings = {**encoded.timings_ms, **found.timings_ms, "encode_ms": encode_ms}
        timings["total_ms"] = encode_ms + float(found.timings_ms.get("search_ms", 0.0))
        return [replace(found, accumulated_audio_sec=len(waveform) / 16000,
                        timings_ms=timings, chunk_id=0, is_final=True)]
    session = AccumulatedAudioRetrievalSession(
        runtime, index, chunk_size_sec=chunk_size_sec, sample_rate=16000, top_k=top_k
    )
    batches = []
    for begin in range(0, len(waveform), feed_samples):
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
    index_metadata = _index_metadata(args.index, args.checkpoint, args.require_index_metadata)
    if args.expected_catalog and index_metadata.get("catalog_sha256") != _sha256(args.expected_catalog):
        raise ValueError("index/catalog SHA256 mismatch; rebuild the index")
    if args.require_target_coverage:
        _validate_targets(records, args.index)
    model, runtime, _processor, _payload = build_glclap_runtime(config, checkpoint=args.checkpoint)
    model.eval()
    index = HotwordEmbeddingIndex.load(args.index)
    index.to(next(model.adapters.parameters()).device)
    stream_cfg = dict(config.get("streaming", {}))
    chunk_size_sec = float(stream_cfg.get("chunk_size_sec", 2.0))
    feed_step_ms = int(stream_cfg.get("feed_step_ms", 100))
    top_k = int(stream_cfg.get("top_k", 50))
    if feed_step_ms <= 0 or chunk_size_sec <= 0 or top_k <= 0:
        raise ValueError("feed_step_ms, chunk_size_sec and top_k must be positive")
    feed_samples = max(1, round(16000 * feed_step_ms / 1000))

    output = Path(args.output)
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
        "mode": args.mode, "resolved_config": config, "versions": versions,
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
            serialized_batches = [batch.to_dict() for batch in batches]
            output_record = {
                "retrieval_mode": args.mode,
                "utt_id": utt_id,
                "audio": manifest_source(record),
                "text": manifest_target(record, required=False),
                "target_hotword_ids": record.get(
                    "target_hotword_ids", record.get("hotword_ids", [])
                ),
                "boundary_group": record.get("boundary_group", "unspecified"),
                "duration_sec": len(waveform) / 16000,
                "elapsed_sec": elapsed,
                "rtf": elapsed / max(1e-9, len(waveform) / 16000),
                "peak_memory_mb": peak_memory,
                "batches": serialized_batches,
                "final_batch": final.to_dict(),
            }
            if offline_exact is not None:
                output_record["offline_final_exact_match"] = offline_exact
            writer.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            writer.flush()
            if trace_dir is not None:
                trace_path = trace_dir / f"{utt_id}.jsonl"
                with trace_path.open("w", encoding="utf-8") as trace:
                    for batch in serialized_batches:
                        trace.write(json.dumps(batch, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
