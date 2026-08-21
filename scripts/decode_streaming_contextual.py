"""Run Qwen3-ASR accumulated-audio streaming with contextual control."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from asr.audio_io import read_wav_mono_float
from asr.backends import QwenVLLMBackend
from asr.config import load_config, require_mapping
from asr.contextual.catalog import HotwordCatalog
from asr.contextual.session import ContextualSessionConfig, ContextualStreamingSession
from asr.contextual.trace import TraceWriter
from asr.data.manifest import manifest_key, manifest_source, manifest_target


def _jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    """Parse the reproducible contextual decoding CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-dir")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    """Decode every PCM16 WAV record in a JSONL manifest."""

    args = parse_args()
    config = load_config(args.config, args.override)
    backend_cfg = dict(require_mapping(config, "backend"))
    session_cfg = ContextualSessionConfig(**dict(require_mapping(config, "session")))
    feed_step_ms = int(config.get("runtime", {}).get("feed_step_ms", 100))
    model = str(backend_cfg.pop("model", "Qwen/Qwen3-ASR-0.6B"))
    language = backend_cfg.pop("language", None)
    backend = QwenVLLMBackend.from_pretrained(model=model, language=language, **backend_cfg)
    catalog = HotwordCatalog.from_jsonl(args.catalog)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as writer:
        for record in _jsonl(args.manifest):
            utt_id = manifest_key(record)
            waveform = read_wav_mono_float(manifest_source(record), session_cfg.sample_rate)
            enabled = record.get("hotword_ids") or list(catalog.entries)
            trace = TraceWriter(trace_dir / f"{utt_id}.jsonl") if trace_dir else None
            try:
                session = ContextualStreamingSession(
                    backend,
                    catalog,
                    session_cfg,
                    enabled_ids=enabled,
                    static_context=str(record.get("context", "")),
                    trace_writer=trace,
                )
                step_samples = max(1, round(session_cfg.sample_rate * feed_step_ms / 1000))
                started = time.perf_counter()
                partials = []
                for position in range(0, len(waveform), step_samples):
                    result = session.step(waveform[position : position + step_samples])
                    if result.chunk_id >= 0:
                        partials.append(result.partial_text)
                final = session.finish()
                elapsed = time.perf_counter() - started
            finally:
                if trace is not None:
                    trace.close()
            output_record = {
                "utt_id": utt_id,
                "reference": manifest_target(record, required=False),
                "hypothesis": final.partial_text,
                "stable_text": final.stable_text,
                "hotwords": [catalog.entries[item].text for item in enabled],
                "boundary_group": record.get("boundary_group", "unspecified"),
                "duration_sec": len(waveform) / session_cfg.sample_rate,
                "elapsed_sec": elapsed,
                "rtf": elapsed / max(1e-9, len(waveform) / session_cfg.sample_rate),
                "revision_rate": final.debug.get("revision_count_total", 0) / max(1, len(final.stable_text)),
                "partials": partials,
            }
            writer.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            writer.flush()


if __name__ == "__main__":
    main()
