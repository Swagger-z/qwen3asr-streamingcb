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
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _load(args.input)
    metrics = evaluate_glclap_records(records)
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
