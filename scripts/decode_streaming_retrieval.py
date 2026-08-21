"""Run standalone accumulated-audio GLCLAP Top-K retrieval over a manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.audio_io import read_wav_mono_float
from asr.config import load_config
from asr.contextual.glclap import AccumulatedAudioRetrievalSession, HotwordEmbeddingIndex
from asr.contextual.glclap_runtime import build_glclap_runtime, jsonl_records
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
    parser.add_argument("--verify-offline", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _same_hits(left: object, right: object) -> bool:
    return left == right


def main() -> None:
    """Write one retrieval record and optional per-refresh trace per utterance."""

    import torch

    args = parse_args()
    config = load_config(args.config, args.override)
    model, runtime, _processor, _payload = build_glclap_runtime(config, checkpoint=args.checkpoint)
    model.eval()
    index = HotwordEmbeddingIndex.load(args.index)
    index.to(next(model.adapters.parameters()).device)
    stream_cfg = dict(config.get("streaming", {}))
    chunk_size_sec = float(stream_cfg.get("chunk_size_sec", 2.0))
    feed_step_ms = int(stream_cfg.get("feed_step_ms", 100))
    top_k = int(stream_cfg.get("top_k", 50))
    feed_samples = max(1, round(16000 * feed_step_ms / 1000))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as writer:
        for record in jsonl_records(args.manifest):
            utt_id = manifest_key(record)
            waveform = read_wav_mono_float(manifest_source(record), 16000)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            session = AccumulatedAudioRetrievalSession(
                runtime,
                index,
                chunk_size_sec=chunk_size_sec,
                sample_rate=16000,
                top_k=top_k,
            )
            batches = []
            started = time.perf_counter()
            for begin in range(0, len(waveform), feed_samples):
                batches.extend(session.step(waveform[begin : begin + feed_samples]))
            final = session.finish()
            if batches and batches[-1].chunk_id == final.chunk_id:
                batches[-1] = final
            else:
                batches.append(final)
            elapsed = time.perf_counter() - started

            offline_exact = None
            if args.verify_offline:
                encoded = runtime.encode_pcm(waveform)
                offline = index.search(encoded.frames, top_k=top_k)
                offline_exact = _same_hits(final.hits, offline.hits)
            peak_memory = None
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            serialized_batches = [batch.to_dict() for batch in batches]
            output_record = {
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
