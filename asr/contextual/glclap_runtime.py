"""Shared construction helpers for the four GLCLAP command-line tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from asr.config import require_mapping

from .glclap_model import (
    GLCLAPRetrieverModel,
    QwenGLCLAPEncoder,
    QwenGLCLAPRuntime,
    load_glclap_checkpoint,
    load_qwen_transformers,
)


def jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSON objects from a UTF-8 JSONL file."""

    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_negative_vocabulary(path: str | Path) -> list[str]:
    """Load negatives from JSONL, one-term-per-line, or ``TERM COUNT`` text."""

    source = Path(path)
    if source.suffix.casefold() == ".jsonl":
        values: list[str] = []
        for record in jsonl_records(source):
            if "text" in record:
                values.append(str(record["text"]))
            aliases = record.get("aliases", ())
            if isinstance(aliases, str):
                aliases = [aliases]
            values.extend(str(value) for value in aliases)
        return sorted({value for value in values if value})
    values: set[str] = set()
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.rsplit(maxsplit=1)
        if len(fields) == 2 and fields[1].isdigit():
            stripped = fields[0].strip()
        if stripped:
            values.add(stripped)
    return sorted(values)


def _device_from_config(config: Mapping[str, Any]) -> str:
    runtime = dict(config.get("runtime", {}))
    requested = str(runtime.get("device", "cuda"))
    try:
        import torch
    except ImportError as exc:
        raise ImportError("GLCLAP runtime requires PyTorch") from exc
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if bool(runtime.get("allow_cpu", False)):
            return "cpu"
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def build_glclap_runtime(
    config: Mapping[str, Any],
    *,
    checkpoint: str | Path | None = None,
) -> tuple[Any, Any, Any, Mapping[str, Any] | None]:
    """Load pinned Qwen, construct the retrieval branch, and place trainables."""

    import torch

    model_cfg = dict(require_mapping(config, "model"))
    model_name = str(model_cfg.get("qwen_model", "Qwen/Qwen3-ASR-0.6B"))
    device = _device_from_config(config)
    load_kwargs: dict[str, Any] = {}
    dtype_name = str(model_cfg.get("qwen_dtype", "bfloat16"))
    if dtype_name == "bfloat16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif dtype_name == "float16":
        load_kwargs["torch_dtype"] = torch.float16
    elif dtype_name == "float32":
        load_kwargs["torch_dtype"] = torch.float32
    else:
        raise ValueError(f"unsupported qwen_dtype: {dtype_name}")
    load_kwargs["device_map"] = device
    qwen_model, processor = load_qwen_transformers(model_name, **load_kwargs)
    payload = None
    if checkpoint is None:
        encoder = QwenGLCLAPEncoder(qwen_model)
        retriever = GLCLAPRetrieverModel(
            encoder,
            mode=str(model_cfg.get("initialization", "qwen_post_projector_frozen")),
            embedding_dim=int(model_cfg.get("embedding_dim", 512)),
            temperature=float(config.get("loss", {}).get("temperature", 0.07)),
        )
    else:
        retriever, payload = load_glclap_checkpoint(checkpoint, qwen_model, map_location="cpu")
    # Move only the small retrieval branch. Qwen is already placed by
    # ``device_map`` and remains frozen/read-only.
    retriever.adapters.to(device=device, dtype=torch.float32)
    retriever.log_temperature.data = retriever.log_temperature.data.to(device)
    if retriever.retrieval_projector is not None:
        retriever.retrieval_projector.to(device=device, dtype=torch.float32)
    runtime = QwenGLCLAPRuntime(retriever, processor, sample_rate=16000)
    return retriever, runtime, processor, payload


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    """Yield deterministic contiguous mini-batches."""

    if size <= 0:
        raise ValueError("batch size must be positive")
    for begin in range(0, len(values), size):
        yield values[begin : begin + size]
