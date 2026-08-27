"""Extract frozen Qwen3-ASR AuT states for phoneme-head training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Run official Transformers inference and capture audio-encoder outputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--language")
    args = parser.parse_args()
    try:
        import torch
        from asr.qwen_compat import import_qwen_symbol

        Qwen3ASRModel = import_qwen_symbol("Qwen3ASRModel")
    except ImportError as exc:
        raise SystemExit("install the 'qwen' and 'probe' extras") from exc
    from asr.contextual.probes import AuTPhonemeProbe, resolve_qwen_audio_encoder

    asr = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    encoder = resolve_qwen_audio_encoder(asr.model)
    captured: list[torch.Tensor] = []

    def hook(module, inputs, output):
        del module, inputs
        hidden = AuTPhonemeProbe._extract_hidden(output)
        captured.append(hidden.detach().float().cpu())

    handle = encoder.register_forward_hook(hook)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Path(args.manifest).open("r", encoding="utf-8") as source, output_manifest.open("w", encoding="utf-8") as target:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                captured.clear()
                asr.transcribe(audio=record["audio"], language=args.language)
                if not captured:
                    raise RuntimeError(f"audio encoder hook produced no state for {record['utt_id']}")
                features = captured[-1]
                if features.ndim == 3:
                    features = features[0]
                feature_path = output_dir / f"{record['utt_id']}.pt"
                torch.save({"features": features, "target_ids": record["target_ids"]}, feature_path)
                target.write(
                    json.dumps(
                        {
                            "utt_id": record["utt_id"],
                            "features": str(feature_path),
                            "target_ids": record["target_ids"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
