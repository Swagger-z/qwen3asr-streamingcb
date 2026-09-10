"""Convert GLCLAP STOP1/STOP2 metadata into project evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import sha256_file, write_jsonl
from asr.contextual.glclap_data import transcript_positive_mask
from asr.contextual.glclap_english import stable_english_id
from asr.data.manifest import manifest_key, manifest_target
from asr.data.manifest_dataset import ManifestDataset


def _mapping_file(path: Path, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected 'UTT_ID {label}'")
            key, value = fields
            if key in result:
                raise ValueError(f"{path}:{line_number}: duplicate ID {key!r}")
            result[key] = value.strip()
    return result


def _rewrite_path(value: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if value.startswith(old):
            return new + value[len(old) :]
    return value


def main() -> None:
    """Create target catalog and retrieval manifest for one STOP split."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--hotlists", required=True)
    parser.add_argument("--hotlists-uniq", required=True)
    parser.add_argument("--dataset", choices=("stop1", "stop2"), required=True)
    parser.add_argument("--transcript-manifest")
    parser.add_argument("--path-prefix", action="append", default=[], help="OLD=NEW")
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--target-catalog-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--allow-missing-audio", action="store_true")
    args = parser.parse_args()
    replacements = []
    for spec in args.path_prefix:
        if "=" not in spec:
            raise ValueError("--path-prefix expects OLD=NEW")
        replacements.append(tuple(spec.split("=", 1)))
    wav_path = Path(args.wav_scp).resolve()
    hotlists_path = Path(args.hotlists).resolve()
    uniq_path = Path(args.hotlists_uniq).resolve()
    audio = {
        key: _rewrite_path(value, replacements)
        for key, value in _mapping_file(wav_path, "AUDIO_PATH").items()
    }
    hotwords: defaultdict[str, list[str]] = defaultdict(list)
    with hotlists_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"{hotlists_path}:{line_number}: expected 'UTT_ID HOTWORD'")
            if fields[1] not in hotwords[fields[0]]:
                hotwords[fields[0]].append(fields[1])
    unique_terms = {
        line.strip() for line in uniq_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    observed_terms = {term for values in hotwords.values() for term in values}
    if not observed_terms <= unique_terms:
        raise ValueError("hotlists contains terms absent from hotlists.uniq")
    transcripts: dict[str, str] = {}
    if args.transcript_manifest:
        transcript_set = ManifestDataset(args.transcript_manifest)
        for record in transcript_set:
            source_key = str(record.get("source_key", manifest_key(record)))
            transcripts[source_key] = manifest_target(record)
    missing_audio = sorted(set(hotwords) - set(audio))
    if missing_audio:
        raise ValueError(f"hotlists IDs absent from wav.scp: {missing_audio[:5]}")
    records = []
    for source_key in sorted(hotwords):
        source = audio[source_key]
        if not args.allow_missing_audio and not Path(source).is_file():
            raise ValueError(f"missing STOP audio for {source_key!r}: {source}")
        transcript = transcripts.get(source_key, "")
        terms = hotwords[source_key]
        if transcript:
            mask = transcript_positive_mask([transcript], terms, language="en")
            missing = [term for term, present in zip(terms, mask[0]) if not present]
            if missing:
                raise ValueError(f"{source_key}: hotwords absent from transcript: {missing}")
        entities = []
        target_ids = []
        for index, term in enumerate(terms):
            hotword_id = stable_english_id(f"en-{args.dataset}", term)
            target_ids.append(hotword_id)
            entities.append(
                {
                    "id": hotword_id,
                    "hotword_id": hotword_id,
                    "mention_id": f"{args.dataset}:{source_key}#hotword-{index:02d}",
                    "text": term,
                    "type": "person" if args.dataset == "stop1" else "location",
                }
            )
        records.append(
            {
                "key": f"{args.dataset}:{source_key}",
                "source_key": source_key,
                "source": source,
                "target": transcript,
                "language": "en",
                "corpus": args.dataset,
                "split": "test",
                "entities": entities,
                "target_hotword_ids": target_ids,
                "metadata": {
                    "source": "GLCLAP STOP1/STOP2",
                    "timing_evaluable": bool(transcript),
                    "retrieval_only": not bool(transcript),
                },
            }
        )
    version = f"{args.dataset}-targets-v1"
    catalog = [
        {
            "catalog_version": version,
            "id": stable_english_id(f"en-{args.dataset}", term),
            "text": term,
            "aliases": [],
            "language": "en",
            "weight": 1.0,
            "metadata": {
                "source": "GLCLAP STOP1/STOP2",
                "role": "evaluation_target",
                "type": "person" if args.dataset == "stop1" else "location",
            },
        }
        for term in sorted(unique_terms)
    ]
    write_jsonl(args.manifest_output, records)
    write_jsonl(args.target_catalog_output, catalog)
    report = {
        "format_version": 1,
        "dataset": args.dataset,
        "utterances": len(records),
        "target_terms": len(catalog),
        "target_mentions": sum(len(row["entities"]) for row in records),
        "timing_evaluable": bool(args.transcript_manifest),
        "manifest_sha256": sha256_file(args.manifest_output),
        "catalog_sha256": sha256_file(args.target_catalog_output),
        "inputs": {
            "wav_scp": {"path": str(wav_path), "sha256": sha256_file(wav_path)},
            "hotlists": {"path": str(hotlists_path), "sha256": sha256_file(hotlists_path)},
            "hotlists_uniq": {"path": str(uniq_path), "sha256": sha256_file(uniq_path)},
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
