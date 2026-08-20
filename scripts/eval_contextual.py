"""Aggregate contextual ASR JSONL metrics and optional paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from asr.eval.contextual_metrics import error_rate, paired_bootstrap_ci
from asr.eval.paper_metrics import evaluate_paper_records


def _load(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    """Evaluate one run and optionally compare it to a baseline by utterance."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    records = _load(args.input)
    metrics = evaluate_paper_records(records)
    if args.baseline:
        baseline_by_id = {str(item["utt_id"]): item for item in _load(args.baseline)}
        differences = []
        for record in records:
            baseline = baseline_by_id.get(str(record["utt_id"]))
            if baseline is None:
                continue
            proposed_error = error_rate(record["reference"], record["hypothesis"], character=True)
            baseline_error = error_rate(baseline["reference"], baseline["hypothesis"], character=True)
            differences.append(baseline_error - proposed_error)
        if differences:
            lower, upper = paired_bootstrap_ci(differences, samples=args.bootstrap_samples)
            metrics["paired_cer_improvement_mean"] = sum(differences) / len(differences)
            metrics["paired_cer_improvement_ci95_low"] = lower
            metrics["paired_cer_improvement_ci95_high"] = upper
    payload = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
