"""JSONL trace output for streaming research instrumentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, TextIO


class TraceWriter:
    """Append and flush one JSON object per processed chunk."""

    def __init__(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = target.open("w", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        """Serialize one trace record immediately."""

        self._stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._stream.flush()

    def close(self) -> None:
        """Close the trace stream."""

        self._stream.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()
