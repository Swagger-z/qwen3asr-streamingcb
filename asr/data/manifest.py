"""Canonical JSONL manifest field accessors.

The project accepts the dataset-facing ``key/source/target`` schema.  Older
internal manifests used ``utt_id/audio/text``; readers keep accepting those
aliases so existing derived artifacts remain usable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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


def validate_manifest_schema(
    records: list[Mapping[str, Any]],
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
