"""Utilities for converting token alignments into hotword-level spans."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from asr.contextual.types import HotwordEntry


@dataclass(frozen=True)
class AlignmentItem:
    """One word/character item returned by an offline forced aligner."""

    text: str
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if self.start_time < 0 or self.end_time < self.start_time:
            raise ValueError(f"invalid alignment interval: {self.start_time}, {self.end_time}")


@dataclass(frozen=True)
class HotwordSpan:
    """Resolved time interval for one catalog spelling."""

    variant: str
    start_time: float
    end_time: float
    first_item: int
    last_item: int


def normalize_alignment_text(text: str) -> str:
    """Normalize text for alignment matching while retaining letters and digits."""

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _flatten_items(items: Sequence[AlignmentItem]) -> tuple[str, tuple[int, ...]]:
    characters: list[str] = []
    item_indices: list[int] = []
    for item_index, item in enumerate(items):
        normalized = normalize_alignment_text(item.text)
        characters.extend(normalized)
        item_indices.extend([item_index] * len(normalized))
    return "".join(characters), tuple(item_indices)


def find_hotword_spans(
    items: Sequence[AlignmentItem],
    variants: Iterable[str],
) -> tuple[HotwordSpan, ...]:
    """Find all distinct contiguous catalog spellings in aligned items."""

    flattened, item_indices = _flatten_items(items)
    if not flattened:
        return ()
    spans: list[HotwordSpan] = []
    seen_intervals: set[tuple[int, int]] = set()
    seen_variants: set[str] = set()
    for raw_variant in variants:
        variant = str(raw_variant)
        normalized = normalize_alignment_text(variant)
        if not normalized or normalized in seen_variants:
            continue
        seen_variants.add(normalized)
        offset = 0
        while True:
            position = flattened.find(normalized, offset)
            if position < 0:
                break
            final_position = position + len(normalized) - 1
            first_item = item_indices[position]
            last_item = item_indices[final_position]
            interval = (first_item, last_item)
            if interval not in seen_intervals:
                seen_intervals.add(interval)
                spans.append(
                    HotwordSpan(
                        variant=variant,
                        start_time=items[first_item].start_time,
                        end_time=items[last_item].end_time,
                        first_item=first_item,
                        last_item=last_item,
                    )
                )
            offset = position + 1
    return tuple(sorted(spans, key=lambda span: (span.start_time, span.end_time, span.variant)))


def resolve_hotword_span(
    items: Sequence[AlignmentItem],
    variants: Iterable[str],
    occurrence_policy: str = "error",
    occurrence_index: int | None = None,
) -> HotwordSpan:
    """Resolve one span, optionally selecting an annotated occurrence index."""

    spans = find_hotword_spans(items, variants)
    if not spans:
        raise ValueError("none of the hotword spellings occur in the aligner output")
    span: HotwordSpan
    if occurrence_index is not None:
        if occurrence_index < 0 or occurrence_index >= len(spans):
            raise ValueError(
                f"annotated occurrence {occurrence_index} is unavailable; "
                f"aligner found {len(spans)} spans"
            )
        span = spans[occurrence_index]
    elif len(spans) > 1:
        if occurrence_policy == "first":
            span = spans[0]
        elif occurrence_policy == "last":
            span = spans[-1]
        elif occurrence_policy != "error":
            raise ValueError(f"unknown occurrence policy: {occurrence_policy}")
        else:
            raise ValueError(f"hotword occurrence is ambiguous: {len(spans)} spans found")
    else:
        span = spans[0]
    if span.end_time <= span.start_time:
        raise ValueError(
            f"hotword has a zero-duration alignment: {span.start_time}, {span.end_time}"
        )
    return span


def record_target_ids(record: Mapping[str, object]) -> tuple[str, ...]:
    """Return unique target IDs from either supported manifest field."""

    raw = record.get("target_hotword_ids", record.get("hotword_ids", ()))
    if isinstance(raw, str):
        raw = (raw,)
    target_ids: list[str] = []
    for value in raw or ():
        hotword_id = str(value)
        if hotword_id and hotword_id not in target_ids:
            target_ids.append(hotword_id)
    return tuple(target_ids)


def record_target_mentions(record: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Return mention-level targets, falling back to one mention per legacy ID."""

    raw_entities = record.get("entities", ())
    if isinstance(raw_entities, Sequence) and not isinstance(raw_entities, (str, bytes)):
        mentions: list[dict[str, object]] = []
        seen_mention_ids: set[str] = set()
        for index, value in enumerate(raw_entities):
            if not isinstance(value, Mapping):
                raise ValueError(f"entities[{index}] must be an object")
            hotword_id = str(value.get("hotword_id", value.get("target_hotword_id", "")))
            if not hotword_id:
                raise ValueError(f"entities[{index}] is missing hotword_id")
            mention_id = str(value.get("mention_id", f"{hotword_id}#{index}"))
            if not mention_id or mention_id in seen_mention_ids:
                raise ValueError(f"duplicate or empty mention_id: {mention_id!r}")
            seen_mention_ids.add(mention_id)
            mention = dict(value)
            mention["mention_id"] = mention_id
            mention["hotword_id"] = hotword_id
            if "occurrence_index" in mention:
                occurrence_index = int(mention["occurrence_index"])
                if occurrence_index < 0:
                    raise ValueError(f"{mention_id}: occurrence_index must be non-negative")
                mention["occurrence_index"] = occurrence_index
            mentions.append(mention)
        if mentions:
            return tuple(mentions)
    return tuple(
        {"mention_id": hotword_id, "hotword_id": hotword_id}
        for hotword_id in record_target_ids(record)
    )


def aligned_records_for_source(
    record: Mapping[str, object],
    entries: Mapping[str, HotwordEntry],
    items: Sequence[AlignmentItem],
    occurrence_policy: str = "error",
) -> tuple[dict[str, object], ...]:
    """Expand one utterance into one aligned record per annotated entity mention."""

    from .manifest import manifest_key, manifest_source
    from .timed_entities import stable_record_key

    source_utt_id = str(record.get("source_utt_id", manifest_key(record)))
    if not source_utt_id:
        raise ValueError("manifest record is missing utt_id")
    mentions = record_target_mentions(record)
    if not mentions:
        raise ValueError(f"{source_utt_id}: no target_hotword_ids")
    all_target_ids = record_target_ids(record)
    aligned_records: list[dict[str, object]] = []
    for mention in mentions:
        hotword_id = str(mention["hotword_id"])
        mention_id = str(mention["mention_id"])
        if hotword_id not in entries:
            raise ValueError(f"{source_utt_id}: unknown target hotword ID {hotword_id!r}")
        entry = entries[hotword_id]
        annotated_text = str(mention.get("text", ""))
        variants = (annotated_text,) if annotated_text else (entry.text, *entry.aliases)
        annotated_occurrence = mention.get("occurrence_index")
        span = resolve_hotword_span(
            items,
            variants,
            occurrence_policy=occurrence_policy,
            occurrence_index=(
                int(annotated_occurrence) if annotated_occurrence is not None else None
            ),
        )
        output = dict(record)
        output["source_utt_id"] = source_utt_id
        output["target_hotword_ids"] = [hotword_id]
        timed_mention = {**mention, "start_sec": round(span.start_time, 3),
                         "end_sec": round(span.end_time, 3)}
        output["entities"] = [timed_mention]
        output["key"] = output["utt_id"] = stable_record_key(source_utt_id, mention_id)
        output["source"] = output["audio"] = manifest_source(record)
        output["focus_mention_id"] = mention_id
        output["aligned_hotword_id"] = hotword_id
        output["aligned_mention_id"] = mention_id
        output["aligned_hotword_variant"] = span.variant
        output["hotword_start_sec"] = round(span.start_time, 3)
        output["hotword_end_sec"] = round(span.end_time, 3)
        aligned_records.append(output)
    timed_entities = [dict(row["entities"][0]) for row in aligned_records]
    for row in aligned_records:
        row["all_entities"] = timed_entities
        row["all_target_hotword_ids"] = list(all_target_ids)
    return tuple(aligned_records)
