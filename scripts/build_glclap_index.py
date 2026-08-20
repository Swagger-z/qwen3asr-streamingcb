"""Build an fp16 GLCLAP index for canonical hotwords and aliases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.config import load_config
from asr.contextual.catalog import HotwordCatalog
from asr.contextual.glclap import HotwordEmbeddingIndex
from asr.contextual.glclap_runtime import build_glclap_runtime


def parse_args() -> argparse.Namespace:
    """Parse index construction arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Encode every catalog spelling once and serialize the exact-search index."""

    import torch

    args = parse_args()
    config = load_config(args.config, args.override)
    catalog = HotwordCatalog.from_jsonl(args.catalog)
    model, runtime, _processor, _payload = build_glclap_runtime(config, checkpoint=args.checkpoint)

    hotword_ids: list[str] = []
    variants: list[str] = []
    for hotword_id, entry in catalog.entries.items():
        seen: set[str] = set()
        for variant in (entry.text, *entry.aliases):
            if variant and variant not in seen:
                seen.add(variant)
                hotword_ids.append(hotword_id)
                variants.append(variant)
    with torch.inference_mode():
        keys = runtime.encode_texts(
            variants,
            batch_size=int(config.get("index", {}).get("text_batch_size", 512)),
        )
    index = HotwordEmbeddingIndex(
        keys,
        hotword_ids,
        variants,
        block_size=int(config.get("index", {}).get("block_size", 16384)),
        catalog_version=catalog.version,
    )
    index.save(args.output)
    metadata = {
        "format_version": index.FORMAT_VERSION,
        "catalog_version": catalog.version,
        "hotword_count": index.hotword_count,
        "variant_count": index.variant_count,
        "embedding_dim": index.embedding_dim,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "catalog_sha256": _sha256(args.catalog),
        "mode": model.mode,
    }
    Path(str(args.output) + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
