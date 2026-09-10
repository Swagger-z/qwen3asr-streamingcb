"""Evaluate GLCLAP retrieval JSONL and optional paired global-only baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.eval.glclap_metrics import (
    evaluate_glclap_records,
    paired_bootstrap_mean_ci,
    record_recall_at_k,
)


def _load(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    """Print and optionally save retrieval, boundary, latency, and Go/No-Go metrics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--baseline", help="Matched global-only CLAP JSONL")
    parser.add_argument(
        "--monolingual-reference",
        help="Matched same-model retrieval against the monolingual catalog",
    )
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--online", action="store_true", help="Require timestamps and validated replay clocks")
    parser.add_argument("--timing-manifest", help="Audited timestamps for fingerprint-matched results")
    parser.add_argument("--deadlines-ms", type=int, nargs="+", default=[0, 100, 200, 500, 1000, 2000])
    parser.add_argument("--entity-output", help="Per-mention/K online JSONL")
    parser.add_argument("--refresh-output", help="Per-refresh eligible-target recall JSONL")
    args = parser.parse_args()

    records = _load(args.input)
    if args.timing_manifest:
        if not args.online:
            raise ValueError("--timing-manifest requires --online")
        from asr.eval.online_retrieval_metrics import attach_entity_timing
        records = attach_entity_timing(records, _load(args.timing_manifest))
    online = None
    if args.online:
        from asr.eval.online_retrieval_metrics import evaluate_online_records
        from asr.contextual.catalog_prep import write_jsonl
        online, entity_rows, refresh_rows = evaluate_online_records(
            records, deadlines_ms=args.deadlines_ms,
            bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        )
        from asr.data.timed_entities import sha256_file
        online["inputs"] = {
            "retrieval": {"path": str(Path(args.input).resolve()), "sha256": sha256_file(args.input)}
        }
        for name, path in (("timing_manifest", args.timing_manifest),
                           ("retrieval_run", str(args.input) + ".run.json")):
            if path and Path(path).is_file():
                online["inputs"][name] = {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
        online["implementation_sha256"] = sha256_file(
            Path(__file__).resolve().parents[1] / "asr/eval/online_retrieval_metrics.py"
        )
        base = args.output or args.input + ".metrics.json"
        entity_path = args.entity_output or base + ".entities.jsonl"
        refresh_path = args.refresh_output or base + ".refreshes.jsonl"
        protected = {
            Path(path).resolve()
            for path in (
                args.input,
                args.timing_manifest,
                args.baseline,
                args.monolingual_reference,
            )
            if path
        }
        destinations = [Path(path).resolve() for path in (entity_path, refresh_path, args.output) if path]
        if len(set(destinations)) != len(destinations) or protected & set(destinations):
            raise ValueError("metric outputs must be distinct from one another and inputs")
        write_jsonl(entity_path, entity_rows)
        write_jsonl(refresh_path, refresh_rows)
    metrics = evaluate_glclap_records(records)
    if online is not None:
        metrics["online"] = online
    gain = None
    ci_low = None
    if args.baseline:
        baseline_by_id = {str(record["utt_id"]): record for record in _load(args.baseline)}
        differences = []
        for record in records:
            baseline = baseline_by_id.get(str(record["utt_id"]))
            if baseline is not None:
                differences.append(record_recall_at_k(record, 50) - record_recall_at_k(baseline, 50))
        if differences:
            gain = sum(differences) / len(differences)
            ci_low, ci_high = paired_bootstrap_mean_ci(
                differences,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            metrics["recall_at_50_gain_vs_global_only"] = gain
            metrics["recall_at_50_gain_ci95_low"] = ci_low
            metrics["recall_at_50_gain_ci95_high"] = ci_high

    if args.monolingual_reference:
        reference_by_id = {
            str(record["utt_id"]): record for record in _load(args.monolingual_reference)
        }
        differences = []
        for record in records:
            reference = reference_by_id.get(str(record["utt_id"]))
            if reference is not None:
                differences.append(
                    record_recall_at_k(reference, 50) - record_recall_at_k(record, 50)
                )
        if differences:
            penalty = sum(differences) / len(differences)
            low, high = paired_bootstrap_mean_ci(
                differences, samples=args.bootstrap_samples, seed=args.seed
            )
            metrics["mixed_catalog_penalty_at_50"] = penalty
            metrics["mixed_catalog_penalty_ci95_low"] = low
            metrics["mixed_catalog_penalty_ci95_high"] = high

    offline_rate = metrics.get("offline_final_exact_match_rate")
    metrics["go_no_go_recall"] = bool(gain is not None and gain >= 0.05 and ci_low is not None and ci_low > 0)
    metrics["go_no_go_boundary"] = bool(
        metrics.get("center_utterances", 0) > 0
        and metrics.get("cross_utterances", 0) > 0
        and metrics.get("boundary_penalty", 1.0) <= 0.03
    )
    metrics["go_no_go_offline_parity"] = bool(offline_rate is not None and offline_rate == 1.0)
    metrics["go_no_go_search_latency"] = bool(metrics.get("search_ms_p95", float("inf")) < 10.0)
    payload = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
