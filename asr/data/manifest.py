"""Canonical JSONL manifest field accessors.

The project accepts the dataset-facing ``key/source/target`` schema.  Older
internal manifests used ``utt_id/audio/text``; readers keep accepting those
aliases so existing derived artifacts remain usable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


SUPPORTED_LANGUAGES = frozenset({"zh", "en"})


def manifest_key(record: Mapping[str, Any]) -> str:
    """Return the utterance key from a canonical or legacy record."""

    for field in ("key", "utt_id"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("manifest record is missing required field 'key'")


def manifest_source(record: Mapping[str, Any]) -> str:
    """Return the audio path from a canonical or legacy record."""

    for field in ("source", "audio"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("manifest record is missing required field 'source'")


def manifest_target(record: Mapping[str, Any], *, required: bool = True) -> str:
    """Return the transcript from a canonical or legacy record."""

    for field in ("target", "text"):
        value = record.get(field)
        if value is not None:
            return str(value)
    if required:
        raise ValueError("manifest record is missing required field 'target'")
    return ""


def manifest_language(
    record: Mapping[str, Any], *, default: str | None = None
) -> str:
    """Return a validated language tag from a manifest record.

    Legacy manifests may omit the field. Callers that intentionally support
    those files can pass ``default="zh"``; strict multilingual pipelines should
    leave *default* unset.
    """

    value = record.get("language", default)
    if value is None or not str(value).strip():
        raise ValueError("manifest record is missing required field 'language'")
    language = str(value).strip().casefold()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"manifest record language must be one of {sorted(SUPPORTED_LANGUAGES)}, "
            f"got {language!r}"
        )
    return language


def manifest_corpus(record: Mapping[str, Any], *, default: str | None = None) -> str:
    """Return the nonempty corpus identifier from a manifest record."""

    value = record.get("corpus", default)
    if value is None or not str(value).strip():
        raise ValueError("manifest record is missing required field 'corpus'")
    return str(value).strip().casefold()


def manifest_split(record: Mapping[str, Any], *, default: str | None = None) -> str:
    """Return the nonempty data split identifier from a manifest record."""

    value = record.get("split", default)
    if value is None or not str(value).strip():
        raise ValueError("manifest record is missing required field 'split'")
    return str(value).strip()


def validate_manifest_schema(
    records: Iterable[Mapping[str, Any]],
    *,
    require_target: bool = True,
) -> None:
    """Validate that every record has the dataset-facing manifest fields."""

    for index, record in enumerate(records, 1):
        try:
            manifest_key(record)
            manifest_source(record)
            manifest_target(record, required=require_target)
        except ValueError as exc:
            raise ValueError(f"manifest record {index}: {exc}") from exc


def validate_multilingual_manifest_schema(
    records: Iterable[Mapping[str, Any]],
    *,
    require_target: bool = True,
    require_namespaced_key: bool = True,
) -> None:
    """Validate the strict dataset-agnostic multilingual training contract."""

    seen: set[str] = set()
    for index, record in enumerate(records, 1):
        try:
            key = manifest_key(record)
            manifest_source(record)
            target = manifest_target(record, required=require_target)
            manifest_language(record)
            corpus = manifest_corpus(record)
            manifest_split(record)
            source_key = str(record.get("source_key", "")).strip()
            if not source_key:
                raise ValueError("manifest record is missing required field 'source_key'")
            if require_target and not target.strip():
                raise ValueError("manifest record target must be non-empty")
            if require_namespaced_key and key != f"{corpus}:{source_key}":
                raise ValueError(
                    "manifest key must equal '<corpus>:<source_key>'; "
                    f"got {key!r} for corpus={corpus!r}, source_key={source_key!r}"
                )
            if key in seen:
                raise ValueError(f"duplicate manifest key {key!r}")
            seen.add(key)
        except ValueError as exc:
            raise ValueError(f"manifest record {index}: {exc}") from exc
