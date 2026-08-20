"""Versioned hotword catalog and path construction."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from .trie import TokenTrie
from .types import HotwordEntry


class TokenizerLike(Protocol):
    """Minimum tokenizer surface required by contextual modules."""

    def encode(self, text: str) -> Sequence[int]:
        """Encode text into integer token IDs."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode integer token IDs into text."""


PronunciationProvider = Callable[[HotwordEntry, str], Iterable[Sequence[str]]]


def normalize_text(text: str) -> str:
    """Apply deterministic matching normalization without language heuristics."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def text_variants(entry: HotwordEntry) -> tuple[str, ...]:
    """Return unique canonical and alias spellings for text matching."""

    variants: list[str] = []
    for value in (entry.text, *entry.aliases):
        for variant in (value, normalize_text(value)):
            if variant and variant not in variants:
                variants.append(variant)
    return tuple(variants)


def tokenizer_variants(entry: HotwordEntry) -> tuple[str, ...]:
    """Return multi-path strings used by token-level biasing."""

    variants: list[str] = []
    for value in text_variants(entry):
        for variant in (value, " " + value, "\n" + value):
            if variant not in variants:
                variants.append(variant)
    return tuple(variants)


def _metadata_pronunciations(entry: HotwordEntry, text: str) -> Iterable[Sequence[str]]:
    raw = entry.metadata.get("pronunciations")
    if raw:
        if isinstance(raw, str):
            raw = [raw]
        for pronunciation in raw:
            if isinstance(pronunciation, str):
                yield tuple(part for part in pronunciation.split() if part)
            else:
                yield tuple(str(part) for part in pronunciation)
        return
    # Deterministic fallback for M1 and tests. Paper experiments should provide
    # pinyin/ARPAbet paths explicitly when constructing the catalog.
    yield tuple(normalize_text(text))


class HotwordCatalog:
    """Immutable-ish global catalog shared by streaming sessions."""

    def __init__(
        self,
        entries: Iterable[HotwordEntry],
        version: str = "1",
        pronunciation_provider: PronunciationProvider | None = None,
    ) -> None:
        self.version = str(version)
        self.entries: dict[str, HotwordEntry] = {}
        self.text_trie = TokenTrie()
        self.pronunciation_trie = TokenTrie()
        provider = pronunciation_provider or _metadata_pronunciations

        ordered = sorted(entries, key=lambda entry: (-entry.weight, entry.id))
        for entry in ordered:
            if entry.id in self.entries:
                raise ValueError(f"duplicate hotword id: {entry.id}")
            self.entries[entry.id] = entry
            for variant in text_variants(entry):
                normalized = normalize_text(variant)
                if normalized:
                    self.text_trie.add(tuple(normalized), entry.id)
            for spelling in (entry.text, *entry.aliases):
                for path in provider(entry, spelling):
                    path_tuple = tuple(path)
                    if path_tuple:
                        self.pronunciation_trie.add(path_tuple, entry.id)
        self.text_trie.build_failure_links()
        self.pronunciation_trie.build_failure_links()

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        pronunciation_provider: PronunciationProvider | None = None,
    ) -> "HotwordCatalog":
        """Load a versioned JSONL catalog."""

        entries: list[HotwordEntry] = []
        version: str | None = None
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_version = str(record.pop("catalog_version", "1"))
                if version is None:
                    version = record_version
                elif version != record_version:
                    raise ValueError(f"mixed catalog versions at line {line_number}")
                aliases = record.get("aliases", ())
                if isinstance(aliases, str):
                    aliases = (aliases,)
                entries.append(
                    HotwordEntry(
                        id=str(record["id"]),
                        text=str(record["text"]),
                        aliases=tuple(str(item) for item in aliases),
                        language=str(record.get("language", "")),
                        weight=float(record.get("weight", 1.0)),
                        metadata=record.get("metadata", {}),
                    )
                )
        return cls(entries, version=version or "1", pronunciation_provider=pronunciation_provider)

    def token_trie(
        self,
        tokenizer: TokenizerLike,
        enabled_ids: Iterable[str] | None = None,
    ) -> TokenTrie:
        """Build a multi-path tokenizer trie for selected entries."""

        selected = set(enabled_ids) if enabled_ids is not None else set(self.entries)
        trie = TokenTrie()
        for hotword_id in sorted(selected, key=lambda item: (-self.entries[item].weight, item)):
            entry = self.entries[hotword_id]
            seen: set[tuple[int, ...]] = set()
            for variant in tokenizer_variants(entry):
                path = tuple(int(token) for token in tokenizer.encode(variant))
                if path and path not in seen:
                    seen.add(path)
                    trie.add(path, hotword_id)
        trie.build_failure_links()
        return trie

    def with_overlay(self, entries: Iterable[HotwordEntry], version: str | None = None) -> "HotwordCatalog":
        """Return a rebuilt catalog containing an overlay of new or replaced IDs."""

        merged = dict(self.entries)
        for entry in entries:
            merged[entry.id] = entry
        return HotwordCatalog(merged.values(), version=version or self.version)

    def weight(self, hotword_id: str) -> float:
        """Return the configured application prior for an entry."""

        return self.entries[hotword_id].weight
