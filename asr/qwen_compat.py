"""Compatibility helpers for loading the pinned ``qwen-asr`` distribution."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def _candidate_source_roots() -> list[Path]:
    """Return likely source roots for an editable qwen-asr installation."""
    roots: list[Path] = []
    for variable in ("QWEN_ASR_SOURCE", "QWEN3_ASR_SOURCE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    try:
        distribution = metadata.distribution("qwen-asr")
    except metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        try:
            direct_url = distribution.read_text("direct_url.json")
            if direct_url:
                url = json.loads(direct_url).get("url", "")
                parsed = urlparse(url)
                if parsed.scheme == "file" and parsed.path:
                    roots.append(Path(unquote(parsed.path)))
        except (OSError, ValueError, TypeError):
            pass
        for item in distribution.files or ():
            if "qwen_asr" in Path(str(item)).parts:
                try:
                    module_file = distribution.locate_file(item)
                    roots.extend((module_file.parent, module_file.parent.parent))
                except (OSError, ValueError):
                    pass
    expanded: list[Path] = []
    for root in roots:
        root = root.resolve()
        expanded.extend((root, root / "src", root / "python"))
    unique: list[Path] = []
    seen: set[str] = set()
    for root in expanded:
        key = os.path.normcase(str(root))
        if key not in seen and root.is_dir():
            seen.add(key)
            unique.append(root)
    return unique


def import_qwen_asr() -> Any:
    """Import qwen_asr, repairing editable-install source path setups."""
    try:
        return importlib.import_module("qwen_asr")
    except ModuleNotFoundError as exc:
        if exc.name != "qwen_asr":
            raise
        original = exc
    for root in _candidate_source_roots():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        importlib.invalidate_caches()
        if importlib.util.find_spec("qwen_asr") is None:
            continue
        try:
            return importlib.import_module("qwen_asr")
        except ModuleNotFoundError as exc:
            if exc.name != "qwen_asr":
                raise
    raise ModuleNotFoundError(
        "The qwen-asr distribution is installed, but qwen_asr is not importable. "
        "Set QWEN_ASR_SOURCE to the qwen3-asr checkout or run "
        "python -m pip install -e /path/to/qwen3-asr."
    ) from original


def import_qwen_symbol(name: str) -> Any:
    """Import a public symbol from qwen_asr with the path compatibility fix."""
    module = import_qwen_asr()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ImportError(f"qwen_asr does not expose {name}; check qwen-asr==0.0.6") from exc
