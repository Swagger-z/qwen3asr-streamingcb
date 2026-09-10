"""Convert AISHELL-2 transcripts and WAV roots to the multilingual JSONL schema."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import sha256_file, write_jsonl


def _audio_index(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*.wav"):
            key = path.stem
            resolved = path.resolve()
            if key in index:
                raise ValueError(f"duplicate AISHELL-2 WAV stem {key!r}: {index[key]} and {resolved}")
            with wave.open(str(resolved), "rb") as reader:
                if reader.getframerate() != 16000:
                    raise ValueError(f"{resolved}: expected 16000 Hz, found {reader.getframerate()}")
                if reader.getnframes() <= 0:
                    raise ValueError(f"{resolved}: empty WAV")
            index[key] = resolved
    if not index:
        raise ValueError("no AISHELL-2 WAV files found")
    return index


def main() -> None:
    """Pair transcript IDs with audio and emit a reproducible audit report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--wav-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    transcript_path = Path(args.transcript).resolve()
    roots = [Path(value).resolve() for value in args.wav_root]
    audio = _audio_index(roots)
    records = []
    seen = set()
    with transcript_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2 or not fields[1].strip():
                raise ValueError(f"{transcript_path}:{line_number}: expected 'UTT_ID TRANSCRIPT'")
            source_key, transcript = fields
            if source_key in seen:
                raise ValueError(f"{transcript_path}:{line_number}: duplicate ID {source_key!r}")
            if source_key not in audio:
                raise ValueError(f"{transcript_path}:{line_number}: missing WAV for {source_key!r}")
            seen.add(source_key)
            records.append(
                {
                    "key": f"aishell2:{source_key}",
                    "source_key": source_key,
                    "source": str(audio[source_key]),
                    "target": transcript.strip(),
                    "language": "zh",
                    "corpus": "aishell2",
                    "split": args.split,
                }
            )
    if not records:
        raise ValueError("empty AISHELL-2 transcript file")
    write_jsonl(args.output, records)
    report = {
        "format_version": 1,
        "source": "AISHELL-2",
        "split": args.split,
        "transcript": str(transcript_path),
        "transcript_sha256": sha256_file(transcript_path),
        "wav_roots": [str(root) for root in roots],
        "indexed_wavs": len(audio),
        "records": len(records),
        "unreferenced_wavs": len(set(audio) - seen),
        "output": str(Path(args.output).resolve()),
        "output_sha256": sha256_file(args.output),
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
