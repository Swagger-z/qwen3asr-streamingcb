"""Convert official AISHELL-NER tagged transcripts into project manifests.

The official flat annotation uses ``[text]`` for PER, ``(text)`` for LOC,
and ``<text>`` for ORG.  This converter never infers entities from a lexicon:
it only parses those gold markers, removes them from the ASR transcript, and
binds utterance IDs to AISHELL-1 WAV files.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.contextual.catalog_prep import normalize_term, sha256_file, stable_catalog_id, write_jsonl


ENTITY_MARKERS = {
    "[": ("]", "PER"),
    "(": (")", "LOC"),
    "<": (">", "ORG"),
}
CLOSING_MARKERS = {closing for closing, _entity_type in ENTITY_MARKERS.values()}


@dataclass(frozen=True)
class EntityMention:
    """One gold entity mention in the marker-free transcript."""

    text: str
    entity_type: str
    char_start: int
    char_end: int
    occurrence_index: int


def parse_annotated_transcript(annotated_text: str) -> tuple[str, tuple[EntityMention, ...]]:
    """Parse one official flat AISHELL-NER transcript without inferring labels."""

    source = unicodedata.normalize("NFKC", str(annotated_text)).strip()
    plain: list[str] = []
    mentions: list[EntityMention] = []
    occurrence_counts: defaultdict[str, int] = defaultdict(int)
    cursor = 0
    while cursor < len(source):
        character = source[cursor]
        marker = ENTITY_MARKERS.get(character)
        if marker is not None:
            closing, entity_type = marker
            closing_index = source.find(closing, cursor + 1)
            if closing_index < 0:
                raise ValueError(f"unclosed {entity_type} marker at character {cursor}")
            raw_entity = source[cursor + 1 : closing_index]
            if any(value in raw_entity for value in (*ENTITY_MARKERS, *CLOSING_MARKERS)):
                raise ValueError("nested or mismatched AISHELL-NER markers are not supported")
            entity = normalize_term(raw_entity)
            if not entity:
                raise ValueError(f"empty {entity_type} entity at character {cursor}")
            char_start = len(plain)
            plain.extend(entity)
            occurrence_index = occurrence_counts[entity]
            occurrence_counts[entity] += 1
            mentions.append(
                EntityMention(
                    text=entity,
                    entity_type=entity_type,
                    char_start=char_start,
                    char_end=char_start + len(entity),
                    occurrence_index=occurrence_index,
                )
            )
            cursor = closing_index + 1
            continue
        if character in CLOSING_MARKERS:
            raise ValueError(f"unmatched closing marker {character!r} at character {cursor}")
        if not character.isspace():
            plain.append(character)
        cursor += 1
    transcript = normalize_term("".join(plain))
    if not transcript:
        raise ValueError("empty transcript after removing AISHELL-NER markers")
    return transcript, tuple(mentions)


def _load_tagged_transcripts(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected 'UTT_ID TAGGED_TRANSCRIPT'")
            utt_id, annotated_text = fields
            if utt_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate utt_id {utt_id!r}")
            seen.add(utt_id)
            records.append((utt_id, annotated_text))
    if not records:
        raise ValueError(f"empty AISHELL-NER transcript file: {path}")
    return records


def _wav_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*.wav"):
        utt_id = path.stem
        if utt_id in index:
            raise ValueError(f"duplicate WAV stem {utt_id!r}: {index[utt_id]} and {path}")
        index[utt_id] = path.resolve()
    if not index:
        raise ValueError(f"no WAV files found under {root}")
    return index


def main() -> None:
    """Materialize gold catalog and all/entity-only manifests from AISHELL-NER."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotated-transcript",
        required=True,
        help="Official data/aishell_ner_transcript.{split}.txt",
    )
    parser.add_argument("--wav-root", required=True, help="AISHELL-1 WAV tree for the split")
    parser.add_argument("--target-catalog-output", required=True)
    parser.add_argument(
        "--eval-manifest-output",
        required=True,
        help="All utterances, including records without an entity",
    )
    parser.add_argument(
        "--entity-manifest-output",
        required=True,
        help="Entity-containing subset consumed by forced alignment/boundary tests",
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    args = parser.parse_args()

    transcript_path = Path(args.annotated_transcript).resolve()
    wav_root = Path(args.wav_root).resolve()
    source_records = _load_tagged_transcripts(transcript_path)
    audio_by_id = _wav_index(wav_root)
    missing_audio = [utt_id for utt_id, _text in source_records if utt_id not in audio_by_id]
    if missing_audio:
        raise ValueError(
            f"missing AISHELL-1 WAV files for {len(missing_audio)} IDs: {missing_audio[:10]}"
        )

    catalog_metadata: defaultdict[str, dict[str, object]] = defaultdict(
        lambda: {"entity_types": set(), "mention_count": 0}
    )
    manifest_records: list[dict[str, object]] = []
    entity_records: list[dict[str, object]] = []
    type_counts: Counter[str] = Counter()
    for utt_id, annotated_text in source_records:
        transcript, parsed_mentions = parse_annotated_transcript(annotated_text)
        entities: list[dict[str, object]] = []
        target_ids: list[str] = []
        for mention_index, mention in enumerate(parsed_mentions):
            hotword_id = stable_catalog_id("aishell-ner", mention.text)
            metadata = catalog_metadata[mention.text]
            entity_types = metadata["entity_types"]
            assert isinstance(entity_types, set)
            entity_types.add(mention.entity_type)
            metadata["mention_count"] = int(metadata["mention_count"]) + 1
            type_counts[mention.entity_type] += 1
            entity = {
                **asdict(mention),
                "mention_id": f"{utt_id}#entity-{mention_index:02d}",
                "hotword_id": hotword_id,
            }
            entities.append(entity)
            if hotword_id not in target_ids:
                target_ids.append(hotword_id)
        record: dict[str, object] = {
            "utt_id": utt_id,
            "audio": str(audio_by_id[utt_id]),
            "text": transcript,
            "target_hotword_ids": target_ids,
            "entities": entities,
            "metadata": {
                "source": "Alibaba-NLP/AISHELL-NER",
                "split": args.split,
                "gold_entity_annotations": True,
            },
        }
        manifest_records.append(record)
        if entities:
            entity_records.append(record)

    catalog_version = f"aishell-ner-{args.split}-targets-v1"
    target_records = []
    for text in sorted(catalog_metadata):
        metadata = catalog_metadata[text]
        entity_types = metadata["entity_types"]
        assert isinstance(entity_types, set)
        target_records.append(
            {
                "catalog_version": catalog_version,
                "id": stable_catalog_id("aishell-ner", text),
                "text": text,
                "aliases": [],
                "language": "zh",
                "weight": 1.0,
                "metadata": {
                    "source": "Alibaba-NLP/AISHELL-NER",
                    "split": args.split,
                    "entity_types": sorted(entity_types),
                    "mention_count": int(metadata["mention_count"]),
                    "role": "evaluation_target",
                },
            }
        )
    if not target_records:
        raise ValueError(f"no entity markers found in {transcript_path}")

    write_jsonl(args.target_catalog_output, target_records)
    write_jsonl(args.eval_manifest_output, manifest_records)
    write_jsonl(args.entity_manifest_output, entity_records)
    report = {
        "format_version": 1,
        "source": "Alibaba-NLP/AISHELL-NER",
        "split": args.split,
        "annotation_semantics": {"[]": "PER", "()": "LOC", "<>": "ORG"},
        "entity_labels_inferred": False,
        "annotated_transcript": str(transcript_path),
        "annotated_transcript_sha256": sha256_file(transcript_path),
        "wav_root": str(wav_root),
        "utterance_count": len(manifest_records),
        "entity_utterance_count": len(entity_records),
        "no_entity_utterance_count": len(manifest_records) - len(entity_records),
        "entity_mention_count": sum(type_counts.values()),
        "unique_entity_count": len(target_records),
        "entity_mention_count_by_type": dict(sorted(type_counts.items())),
        "target_catalog_output": str(Path(args.target_catalog_output).resolve()),
        "eval_manifest_output": str(Path(args.eval_manifest_output).resolve()),
        "entity_manifest_output": str(Path(args.entity_manifest_output).resolve()),
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
