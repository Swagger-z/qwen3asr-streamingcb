"""Deterministic preparation of Chinese GLCLAP hotword catalogs.

The helpers in this module deliberately keep corpus word-frequency lists,
training negatives, and evaluation target entities as separate concepts.
Word-frequency entries are suitable distractors, while gold entity records
must come from an annotated evaluation set.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .glclap_data import batch_negative_exclusions, compact_transcript
from asr.data.manifest import manifest_target


_HAN_TERM = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


@dataclass(frozen=True)
class WordFrequencyEntry:
    """One normalized term with frequency counts from one or more corpora."""

    text: str
    source_frequencies: tuple[tuple[str, int], ...]

    @property
    def total_frequency(self) -> int:
        """Return the summed count across all source corpora."""

        return sum(count for _, count in self.source_frequencies)


@dataclass(frozen=True)
class WordFrequencyStats:
    """Parsing statistics for one source word-frequency file."""

    source: str
    path: str
    sha256: str
    total_lines: int
    accepted_lines: int
    blank_or_comment_lines: int
    filtered_lines: int
    unique_terms: int


def normalize_term(text: str) -> str:
    """NFKC-normalize a Chinese catalog term and remove whitespace."""

    return "".join(unicodedata.normalize("NFKC", str(text)).split())


def is_chinese_term(text: str, *, min_chars: int = 2, max_chars: int = 8) -> bool:
    """Return whether *text* is an all-Han term within the configured length."""

    return min_chars <= len(text) <= max_chars and _HAN_TERM.fullmatch(text) is not None


def parse_word_frequency_line(
    line: str,
    *,
    source: str,
    line_number: int,
) -> tuple[str, int] | None:
    """Parse ``TERM COUNT`` while reporting malformed source locations."""

    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fields = stripped.rsplit(maxsplit=1)
    if len(fields) != 2:
        raise ValueError(f"{source}:{line_number}: expected 'TERM COUNT'")
    term = normalize_term(fields[0])
    try:
        frequency = int(fields[1])
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: invalid frequency {fields[1]!r}") from exc
    if not term or frequency <= 0:
        raise ValueError(f"{source}:{line_number}: term must be non-empty and frequency positive")
    return term, frequency


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_word_frequency_file(
    path: str | Path,
    *,
    source_name: str,
    min_chars: int = 2,
    max_chars: int = 8,
) -> tuple[dict[str, int], WordFrequencyStats]:
    """Load, normalize, filter, and merge duplicate rows from one corpus."""

    source_path = Path(path)
    counts: dict[str, int] = {}
    total_lines = 0
    accepted_lines = 0
    blank_or_comment_lines = 0
    filtered_lines = 0
    with source_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            total_lines += 1
            parsed = parse_word_frequency_line(
                line,
                source=str(source_path),
                line_number=line_number,
            )
            if parsed is None:
                blank_or_comment_lines += 1
                continue
            term, frequency = parsed
            if not is_chinese_term(term, min_chars=min_chars, max_chars=max_chars):
                filtered_lines += 1
                continue
            counts[term] = counts.get(term, 0) + frequency
            accepted_lines += 1
    return counts, WordFrequencyStats(
        source=source_name,
        path=str(source_path),
        sha256=sha256_file(source_path),
        total_lines=total_lines,
        accepted_lines=accepted_lines,
        blank_or_comment_lines=blank_or_comment_lines,
        filtered_lines=filtered_lines,
        unique_terms=len(counts),
    )


def merge_word_frequency_sources(
    sources: Sequence[tuple[str, str | Path]],
    *,
    min_chars: int = 2,
    max_chars: int = 8,
) -> tuple[tuple[WordFrequencyEntry, ...], tuple[WordFrequencyStats, ...]]:
    """Merge named word-frequency files without losing per-source counts."""

    if not sources:
        raise ValueError("at least one word-frequency source is required")
    names = [name for name, _ in sources]
    if len(names) != len(set(names)):
        raise ValueError("word-frequency source names must be unique")
    merged: dict[str, dict[str, int]] = {}
    reports: list[WordFrequencyStats] = []
    for source_name, path in sources:
        counts, report = load_word_frequency_file(
            path,
            source_name=source_name,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        reports.append(report)
        for term, frequency in counts.items():
            merged.setdefault(term, {})[source_name] = frequency
    entries = tuple(
        WordFrequencyEntry(text=term, source_frequencies=tuple(sorted(frequencies.items())))
        for term, frequencies in sorted(merged.items())
    )
    return entries, tuple(reports)


def select_frequency_stratified(
    entries: Sequence[WordFrequencyEntry],
    *,
    limit: int,
    excluded_terms: Iterable[str] = (),
    seed: int = 42,
    bucket_count: int = 10,
) -> tuple[WordFrequencyEntry, ...]:
    """Sample evenly across frequency-rank buckets with deterministic shuffling."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return ()
    excluded: set[str] = set()
    for value in excluded_terms:
        normalized = normalize_term(value)
        if normalized:
            excluded.add(normalized)
    candidates = [entry for entry in entries if entry.text not in excluded]
    if len(candidates) < limit:
        raise ValueError(
            f"need {limit} catalog terms after exclusion, found {len(candidates)}"
        )
    ranked = sorted(candidates, key=lambda item: (-item.total_frequency, item.text))
    bucket_count = max(1, min(int(bucket_count), len(ranked)))
    buckets: list[list[WordFrequencyEntry]] = [[] for _ in range(bucket_count)]
    for rank, entry in enumerate(ranked):
        bucket_index = min(bucket_count - 1, rank * bucket_count // len(ranked))
        buckets[bucket_index].append(entry)
    for bucket_index, bucket in enumerate(buckets):
        random.Random(f"{seed}:{bucket_index}").shuffle(bucket)
    selected: list[WordFrequencyEntry] = []
    offsets = [0] * bucket_count
    while len(selected) < limit:
        progressed = False
        for bucket_index, bucket in enumerate(buckets):
            if offsets[bucket_index] >= len(bucket):
                continue
            selected.append(bucket[offsets[bucket_index]])
            offsets[bucket_index] += 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:  # pragma: no cover - guarded by the size check above
            raise RuntimeError("frequency-stratified sampler exhausted unexpectedly")
    return tuple(sorted(selected, key=lambda item: (-item.total_frequency, item.text)))


def stable_catalog_id(prefix: str, text: str) -> str:
    """Return a stable content-addressed catalog ID."""

    digest = hashlib.sha256(normalize_term(text).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _distractor_record(
    entry: WordFrequencyEntry,
    *,
    version: str,
    id_prefix: str,
    role: str,
) -> dict[str, Any]:
    return {
        "catalog_version": version,
        "id": stable_catalog_id(id_prefix, entry.text),
        "text": entry.text,
        "aliases": [],
        "language": "zh",
        "weight": 1.0,
        "metadata": {
            "role": role,
            "total_frequency": entry.total_frequency,
            "source_frequencies": dict(entry.source_frequencies),
        },
    }


def build_negative_catalog(
    entries: Sequence[WordFrequencyEntry],
    *,
    size: int = 10_000,
    excluded_terms: Iterable[str] = (),
    seed: int = 42,
    version: str = "zh-train-neg-v1",
) -> list[dict[str, Any]]:
    """Build a versioned frequency-stratified training-negative catalog."""

    selected = select_frequency_stratified(
        entries,
        limit=size,
        excluded_terms=excluded_terms,
        seed=seed,
    )
    return [
        _distractor_record(entry, version=version, id_prefix="neg", role="train_negative")
        for entry in selected
    ]


def target_spellings(records: Iterable[Mapping[str, Any]]) -> set[str]:
    """Return normalized canonical and alias spellings from target records."""

    spellings: set[str] = set()
    for record in records:
        aliases = record.get("aliases", ())
        if isinstance(aliases, str):
            aliases = [aliases]
        for value in (record.get("text", ""), *aliases):
            normalized = normalize_term(str(value))
            if normalized:
                spellings.add(normalized)
    return spellings


def build_evaluation_catalog(
    target_records: Sequence[Mapping[str, Any]],
    distractor_entries: Sequence[WordFrequencyEntry],
    *,
    size: int = 10_000,
    evaluation_transcripts: Iterable[str] = (),
    seed: int = 42,
    min_chars: int = 2,
    max_chars: int = 8,
    version: str = "aishell1-ne-10k-v1",
) -> list[dict[str, Any]]:
    """Combine annotated targets with clean, unspoken distractors up to *size*."""

    if not target_records:
        raise ValueError("evaluation catalog requires at least one target record")
    if len(target_records) > size:
        raise ValueError(f"{len(target_records)} targets exceed requested catalog size {size}")
    normalized_targets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    spelling_owner: dict[str, str] = {}
    for raw in target_records:
        if "id" not in raw or "text" not in raw:
            raise ValueError("every target record requires id and text")
        hotword_id = str(raw["id"])
        if hotword_id in seen_ids:
            raise ValueError(f"duplicate target id: {hotword_id}")
        seen_ids.add(hotword_id)
        aliases = raw.get("aliases", ())
        if isinstance(aliases, str):
            aliases = [aliases]
        canonical = normalize_term(str(raw["text"]))
        if not canonical:
            raise ValueError(f"target {hotword_id!r} has an empty canonical text")
        normalized_aliases = [normalize_term(str(value)) for value in aliases]
        normalized_aliases = [value for value in normalized_aliases if value and value != canonical]
        for spelling in (canonical, *normalized_aliases):
            owner = spelling_owner.get(spelling)
            if owner is not None and owner != hotword_id:
                raise ValueError(
                    f"target spelling {spelling!r} is shared by IDs {owner!r} and {hotword_id!r}"
                )
            spelling_owner[spelling] = hotword_id
        metadata = dict(raw.get("metadata", {}))
        metadata["role"] = "target"
        normalized_targets.append(
            {
                "catalog_version": version,
                "id": hotword_id,
                "text": canonical,
                "aliases": sorted(set(normalized_aliases)),
                "language": str(raw.get("language", "zh")),
                "weight": float(raw.get("weight", 1.0)),
                "metadata": metadata,
            }
        )
    transcript_exclusions = batch_negative_exclusions(
        evaluation_transcripts,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    exclusions = set(spelling_owner) | transcript_exclusions
    selected = select_frequency_stratified(
        distractor_entries,
        limit=size - len(normalized_targets),
        excluded_terms=exclusions,
        seed=seed,
    )
    distractors = [
        _distractor_record(
            entry,
            version=version,
            id_prefix="dist",
            role="evaluation_distractor",
        )
        for entry in selected
    ]
    return sorted(normalized_targets, key=lambda item: item["id"]) + distractors


def validate_manifest_targets(
    records: Iterable[Mapping[str, Any]],
    target_records: Iterable[Mapping[str, Any]],
) -> None:
    """Fail if an evaluation manifest references an unknown target hotword ID."""

    known = {str(record["id"]) for record in target_records}
    for line_number, record in enumerate(records, 1):
        values = record.get("target_hotword_ids", record.get("hotword_ids", ()))
        if isinstance(values, str):
            values = [values]
        unknown = {str(value) for value in values} - known
        if unknown:
            raise ValueError(
                f"evaluation manifest record {line_number} references unknown targets: "
                f"{sorted(unknown)}"
            )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load UTF-8 JSON objects with source-aware error reporting."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write deterministic UTF-8 JSONL and return the number of records."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def transcript_texts(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Extract normalized non-empty transcripts from manifest records."""

    texts: list[str] = []
    for record in records:
        normalized = compact_transcript(manifest_target(record, required=False))
        if normalized:
            texts.append(normalized)
    return texts
