"""Dependency-light configuration loading and dotted overrides."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value == "":
        return {}
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("\"'")


def _simple_yaml(text: str) -> dict[str, Any]:
    """Parse the mapping/list subset used by repository configs.

    PyYAML is preferred when installed. This fallback intentionally rejects
    advanced YAML constructs instead of interpreting them inconsistently.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        content = raw.split(" #", 1)[0].rstrip()
        indent = len(content) - len(content.lstrip(" "))
        if "\t" in content[:indent]:
            raise ValueError(f"tabs are not supported in YAML at line {line_number}")
        stripped = content.strip()
        if ":" not in stripped:
            raise ValueError(f"expected mapping entry at line {line_number}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"invalid indentation at line {line_number}")
        parent = stack[-1][1]
        parsed = _scalar(value)
        parent[key] = parsed
        if parsed == {} and not value.strip():
            stack.append((indent, parsed))
    return root


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load JSON/YAML and apply ``dotted.key=value`` overrides."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        config = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            config = _simple_yaml(text)
        else:
            config = yaml.safe_load(text) or {}
    if not isinstance(config, dict):
        raise ValueError("top-level config must be a mapping")
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value: {override}")
        dotted, raw_value = override.split("=", 1)
        set_dotted(config, dotted, _scalar(raw_value))
    return config


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested mapping value, creating intermediate mappings."""

    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError("override key must be non-empty")
    current = config
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot descend into non-mapping key: {part}")
        current = child
    current[parts[-1]] = value


def require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required mapping section with a useful error."""

    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config section '{key}' must be a mapping")
    return value
