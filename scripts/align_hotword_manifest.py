"""Align evaluation transcripts and emit one hotword timestamp span per record."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog import HotwordCatalog
from asr.data.forced_alignment import (
    AlignmentItem,
    aligned_records_for_source,
    record_target_ids,
    record_target_mentions,
)
from asr.data.manifest import manifest_key, manifest_source, manifest_target


LANGUAGE_NAMES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "english": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
}


def parse_args() -> argparse.Namespace:
    """Parse forced-alignment CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Original evaluation JSONL")
    parser.add_argument("--catalog", required=True, help="Annotated target hotword catalog")
    parser.add_argument("--output", required=True, help="Aligned output JSONL for stage2")
    parser.add_argument("--report", required=True, help="Alignment summary JSON")
    parser.add_argument("--trace-output", help="Optional full item-level alignment JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--occurrence-policy", choices=("error", "first", "last"), default="error")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--on-error", choices=("fail", "skip"), default="fail")
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field, accessor in (("key", manifest_key), ("source", manifest_source), ("target", manifest_target)):
                try:
                    accessor(record)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: missing {field}") from exc
            if not record_target_ids(record):
                raise ValueError(f"{path}:{line_number}: missing target_hotword_ids")
            records.append(record)
    if not records:
        raise ValueError(f"empty manifest: {path}")
    return records


def _resolve_audio(record: dict[str, Any], manifest_dir: Path) -> str:
    value = manifest_source(record)
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"audio does not exist: {path}")
    return str(path)


def _language(record: dict[str, Any], default: str) -> str:
    raw = str(record.get("language", default)).strip()
    return LANGUAGE_NAMES.get(raw.casefold(), raw)


def _load_completed(path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            source_id = str(record.get("source_utt_id", record.get("utt_id", "")))
            hotword_id = str(record.get("aligned_hotword_id", ""))
            mention_id = str(record.get("aligned_mention_id", hotword_id))
            if not source_id or not hotword_id or not mention_id:
                raise ValueError(f"{path}:{line_number}: cannot resume malformed aligned record")
            key = (source_id, mention_id)
            if key in completed:
                raise ValueError(f"{path}:{line_number}: duplicate aligned key {key}")
            completed.add(key)
    return completed


def _alignment_items(result: Any) -> tuple[AlignmentItem, ...]:
    raw_items: Iterable[Any] = getattr(result, "items", result)
    return tuple(
        AlignmentItem(
            text=str(getattr(item, "text", item["text"] if isinstance(item, dict) else "")),
            start_time=float(
                getattr(item, "start_time", item["start_time"] if isinstance(item, dict) else 0.0)
            ),
            end_time=float(
                getattr(item, "end_time", item["end_time"] if isinstance(item, dict) else 0.0)
            ),
        )
        for item in raw_items
    )


def _load_aligner(args: argparse.Namespace) -> Any:
    import torch
    from asr.qwen_compat import import_qwen_symbol

    Qwen3ForcedAligner = import_qwen_symbol("Qwen3ForcedAligner")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": args.device}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    return Qwen3ForcedAligner.from_pretrained(args.model, **kwargs)


def _chunks(values: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def main() -> None:
    """Run batched Qwen3 forced alignment with resumable JSONL output."""

    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    manifest_path = Path(args.manifest).resolve()
    catalog_path = Path(args.catalog).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    trace_path = Path(args.trace_output).resolve() if args.trace_output else None
    if output_path.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"output exists; use --resume or --overwrite: {output_path}")

    records = _read_manifest(manifest_path)
    catalog = HotwordCatalog.from_jsonl(catalog_path)
    for record in records:
        unknown = set(record_target_ids(record)) - set(catalog.entries)
        if unknown:
            raise ValueError(f"{manifest_key(record)}: unknown target IDs: {sorted(unknown)}")
        record["audio"] = _resolve_audio(record, manifest_path.parent)

    completed = _load_completed(output_path) if args.resume else set()
    pending = [
        record
        for record in records
        if any(
            (manifest_key(record), str(mention["mention_id"])) not in completed
            for mention in record_target_mentions(record)
        )
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and output_path.exists() else "w"
    trace_mode = "a" if args.resume and trace_path and trace_path.exists() else "w"
    failures: list[dict[str, str]] = []
    written = 0
    started = time.perf_counter()
    aligner = _load_aligner(args) if pending else None

    trace_stream = trace_path.open(trace_mode, encoding="utf-8") if trace_path else None
    try:
        with output_path.open(mode, encoding="utf-8") as output_stream:
            for batch_number, batch in enumerate(_chunks(pending, args.batch_size), 1):
                assert aligner is not None
                results = aligner.align(
                    audio=[str(record["audio"]) for record in batch],
                    text=[manifest_target(record) for record in batch],
                    language=[_language(record, args.language) for record in batch],
                )
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"aligner returned {len(results)} results for a batch of {len(batch)}"
                    )
                for record, result in zip(batch, results):
                    source_id = manifest_key(record)
                    try:
                        items = _alignment_items(result)
                        if not items:
                            raise ValueError("aligner returned no timestamp items")
                        for aligned in aligned_records_for_source(
                            record,
                            catalog.entries,
                            items,
                            occurrence_policy=args.occurrence_policy,
                        ):
                            key = (source_id, str(aligned["aligned_mention_id"]))
                            if key in completed:
                                continue
                            output_stream.write(json.dumps(aligned, ensure_ascii=False) + "\n")
                            output_stream.flush()
                            completed.add(key)
                            written += 1
                        if trace_stream:
                            trace_stream.write(
                                json.dumps(
                                    {
                                        "utt_id": source_id,
                                        "items": [
                                            {
                                                "text": item.text,
                                                "start_time": item.start_time,
                                                "end_time": item.end_time,
                                            }
                                            for item in items
                                        ],
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            trace_stream.flush()
                    except Exception as error:
                        failure = {"utt_id": source_id, "error": str(error)}
                        failures.append(failure)
                        if args.on_error == "fail":
                            raise RuntimeError(f"alignment post-processing failed: {failure}") from error
                print(
                    json.dumps(
                        {
                            "batch": batch_number,
                            "source_records_done": min(batch_number * args.batch_size, len(pending)),
                            "source_records_total": len(pending),
                            "aligned_records_written": written,
                            "failures": len(failures),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        if trace_stream:
            trace_stream.close()

    report = {
        "format_version": 1,
        "model": args.model,
        "qwen_asr_version": importlib.metadata.version("qwen-asr"),
        "manifest": str(manifest_path),
        "catalog": str(catalog_path),
        "output": str(output_path),
        "source_record_count": len(records),
        "pending_source_record_count": len(pending),
        "aligned_key_count": len(completed),
        "new_aligned_record_count": written,
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": args.device,
        "occurrence_policy": args.occurrence_policy,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
