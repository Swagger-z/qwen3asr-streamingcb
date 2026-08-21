"""Build controlled center/cross-boundary PCM16 WAV variants from JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from asr.data.boundary_stress import build_variants
from asr.data.manifest import manifest_key, manifest_source


def main() -> None:
    """Create shifted WAVs using offline reference hotword timestamps."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--chunk-ms", type=int, default=2000)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open("r", encoding="utf-8") as source, output_manifest.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            variants = build_variants(
                manifest_source(record),
                output_dir / manifest_key(record),
                float(record["hotword_start_sec"]),
                float(record["hotword_end_sec"]),
                args.chunk_ms / 1000,
            )
            for variant in variants:
                target.write(json.dumps({**record, **variant}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
