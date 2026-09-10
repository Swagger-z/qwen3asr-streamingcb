"""Collect per-run multilingual retrieval metrics into one experiment summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import sha256_file


def main() -> None:
    """Summarize frozen/global-only, dataset, mode, and catalog-interference results."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir).resolve()
    runs = []
    for path in sorted(root.glob("*.mixed.jsonl.metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        suffix = ".mixed.jsonl.metrics.json"
        stem = path.name[: -len(suffix)]
        parts = stem.split(".")
        if len(parts) < 3:
            continue
        variant, mode = parts[0], parts[-1]
        dataset = ".".join(parts[1:-1])
        runs.append(
            {
                "variant": variant,
                "dataset": dataset,
                "mode": mode,
                "recall_at_50": payload.get("recall_at_50"),
                "mrr": payload.get("mrr"),
                "mixed_catalog_penalty_at_50": payload.get("mixed_catalog_penalty_at_50"),
                "gain_vs_global_only_at_50": payload.get("recall_at_50_gain_vs_global_only"),
                "online": payload.get("online"),
                "metrics_path": str(path),
                "metrics_sha256": sha256_file(path),
            }
        )
    if not runs:
        raise ValueError(f"no multilingual metric files found under {root}")
    output = {
        "format_version": 1,
        "purpose": "multilingual_glclap_experiment_summary",
        "results_dir": str(root),
        "run_count": len(runs),
        "runs": runs,
    }
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
