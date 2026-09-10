"""Indexed, read-only JSONL datasets for large multilingual manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .manifest import (
    manifest_corpus,
    manifest_key,
    manifest_language,
    manifest_source,
    manifest_split,
    manifest_target,
)


class ManifestDataset(Sequence[dict[str, Any]]):
    """Random-access JSONL dataset backed by byte offsets.

    The file is scanned once to validate JSON objects and collect small routing
    metadata. Full records are decoded only when indexed, so million-row
    manifests do not require an in-memory dictionary for every utterance.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        require_target: bool = True,
        strict_multilingual: bool = False,
    ) -> None:
        self.path = Path(path).resolve()
        self.require_target = bool(require_target)
        self.strict_multilingual = bool(strict_multilingual)
        self._offsets: list[int] = []
        self._keys: list[str] = []
        self._languages: list[str] = []
        self._corpora: list[str] = []
        self._splits: list[str] = []
        grouped: defaultdict[str, list[int]] = defaultdict(list)
        grouped_dataset: defaultdict[str, list[int]] = defaultdict(list)
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                digest.update(raw)
                line_number += 1
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{self.path}:{line_number}: invalid JSON object") from exc
                if not isinstance(record, Mapping):
                    raise ValueError(f"{self.path}:{line_number}: JSONL row must be an object")
                try:
                    key = manifest_key(record)
                    manifest_source(record)
                    target = manifest_target(record, required=require_target)
                    if require_target and not target.strip():
                        raise ValueError("manifest target must be non-empty")
                    if strict_multilingual:
                        language = manifest_language(record)
                        corpus = manifest_corpus(record)
                        split = manifest_split(record)
                        source_key = str(record.get("source_key", "")).strip()
                        if not source_key:
                            raise ValueError("manifest record is missing required field 'source_key'")
                        if key != f"{corpus}:{source_key}":
                            raise ValueError("manifest key must equal '<corpus>:<source_key>'")
                    else:
                        language = manifest_language(record, default="zh")
                        corpus = manifest_corpus(record, default="legacy")
                        split = manifest_split(record, default="train")
                except ValueError as exc:
                    raise ValueError(f"{self.path}:{line_number}: {exc}") from exc
                index = len(self._offsets)
                self._offsets.append(offset)
                self._keys.append(key)
                self._languages.append(language)
                self._corpora.append(corpus)
                self._splits.append(split)
                grouped[corpus].append(index)
                grouped_dataset[f"{corpus}/{split}"].append(index)
        duplicates = [key for key, count in Counter(self._keys).items() if count > 1]
        if duplicates:
            raise ValueError(f"{self.path}: duplicate manifest keys: {duplicates[:5]}")
        self._indices_by_corpus = {name: tuple(values) for name, values in grouped.items()}
        self._indices_by_dataset = {
            name: tuple(values) for name, values in grouped_dataset.items()
        }
        self.sha256 = digest.hexdigest()

    @property
    def keys(self) -> tuple[str, ...]:
        """Return manifest keys in file order."""

        return tuple(self._keys)

    @property
    def languages(self) -> tuple[str, ...]:
        """Return one normalized language tag per record."""

        return tuple(self._languages)

    @property
    def corpora(self) -> tuple[str, ...]:
        """Return one normalized corpus tag per record."""

        return tuple(self._corpora)

    @property
    def splits(self) -> tuple[str, ...]:
        """Return one split tag per record."""

        return tuple(self._splits)

    @property
    def indices_by_corpus(self) -> Mapping[str, tuple[int, ...]]:
        """Return immutable corpus-to-row-index groups used by samplers."""

        return self._indices_by_corpus

    @property
    def indices_by_dataset(self) -> Mapping[str, tuple[int, ...]]:
        """Return ``corpus/split`` groups for exact proportional scheduling."""

        return self._indices_by_dataset

    def __len__(self) -> int:
        """Return the number of nonblank JSONL records."""

        return len(self._offsets)

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        """Decode one record or a slice without materializing the whole file."""

        if isinstance(index, slice):
            return self.records_at(range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.records_at((index,))[0]

    def records_at(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Decode several indexed rows while holding one short-lived file handle."""

        normalized: list[int] = []
        for index in indices:
            value = int(index)
            if value < 0:
                value += len(self)
            if value < 0 or value >= len(self):
                raise IndexError(index)
            normalized.append(value)
        records: list[dict[str, Any]] = []
        with self.path.open("rb") as stream:
            for index in normalized:
                stream.seek(self._offsets[index])
                record = json.loads(stream.readline().decode("utf-8"))
                if not isinstance(record, dict):  # pragma: no cover - constructor validates.
                    raise ValueError("JSONL row must be an object")
                records.append(record)
        return records

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield decoded records in file order."""

        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):  # pragma: no cover - constructor validates.
                        raise ValueError("JSONL row must be an object")
                    yield record

    def protocol_summary(self) -> dict[str, Any]:
        """Return stable, JSON-serializable provenance and composition metadata."""

        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "records": len(self),
            "corpus_counts": dict(sorted(Counter(self._corpora).items())),
            "dataset_counts": dict(
                sorted(
                    Counter(
                        f"{corpus}/{split_name}"
                        for corpus, split_name in zip(self._corpora, self._splits)
                    ).items()
                )
            ),
            "language_counts": dict(sorted(Counter(self._languages).items())),
            "split_counts": dict(sorted(Counter(self._splits).items())),
            "indexed_jsonl": True,
        }
