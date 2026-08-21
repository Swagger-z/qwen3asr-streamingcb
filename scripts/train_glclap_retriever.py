"""Train GLCLAP audio/text adapters on frozen Qwen3-ASR-0.6B features."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from asr.audio_io import read_wav_mono_float
from asr.config import load_config, require_mapping
from asr.contextual.glclap_data import (
    batch_negative_exclusions,
    deterministic_local_positive,
    equality_positive_mask,
    sample_shared_negatives,
)
from asr.contextual.glclap_loss import glclap_loss
from asr.contextual.glclap_model import save_glclap_checkpoint
from asr.contextual.glclap_runtime import (
    batched,
    build_glclap_runtime,
    jsonl_records,
    load_negative_vocabulary,
)


def parse_args() -> argparse.Namespace:
    """Parse the reproducible training command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, help="AISHELL-1 JSONL: utt_id/audio/text")
    parser.add_argument("--negative-catalog", required=True, help="JSONL, one term per line, or TERM COUNT text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _extract_qwen_features(model: Any, processor: Any, waveform: Sequence[float]) -> Any:
    import torch

    batch = processor.feature_extractor(
        [np.asarray(waveform, dtype=np.float32)],
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    device = next(model.adapters.parameters()).device
    qwen_dtype = next(model.encoder.audio_tower.parameters()).dtype
    features = batch["input_features"][0].to(device=device, dtype=qwen_dtype)
    feature_len = batch["attention_mask"][0].to(device).sum().long().unsqueeze(0)
    return model.encoder.extract_audio_features(features, feature_len)


def _pad_audio(frames: list[Any]) -> tuple[Any, Any]:
    import torch

    if not frames:
        raise ValueError("cannot pad an empty audio batch")
    lengths = [item.shape[0] for item in frames]
    padded = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)
    positions = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0)
    mask = positions < torch.tensor(lengths, device=padded.device).unsqueeze(1)
    return padded, mask


def _token_keys(model: Any, processor: Any, texts: list[str]) -> Any:
    tokens = processor.tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.adapters.parameters()).device
    return model.encode_text_tokens(
        tokens["input_ids"].to(device),
        tokens["attention_mask"].to(device),
    )


def main() -> None:
    """Run deterministic global/local contrastive adapter training."""

    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required; install the 'probe' and 'qwen' extras") from exc

    config = load_config(args.config, args.override)
    train_cfg = dict(require_mapping(config, "training"))
    loss_cfg = dict(require_mapping(config, "loss"))
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    records = jsonl_records(args.manifest)
    if not records:
        raise ValueError("training manifest is empty")
    for field in ("utt_id", "audio", "text"):
        if any(field not in record for record in records):
            raise ValueError(f"every training record must contain {field!r}")
    negative_vocabulary = load_negative_vocabulary(args.negative_catalog)
    model, _runtime, processor, payload = build_glclap_runtime(config, checkpoint=args.resume)
    device = next(model.adapters.parameters()).device

    adapter_lr = float(train_cfg.get("adapter_lr", 3e-4))
    projector_lr = float(train_cfg.get("projector_lr", 3e-5))
    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(adapter_lr, projector_lr),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    micro_batch = int(train_cfg.get("micro_batch_size", 8))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 48))
    epochs = int(train_cfg.get("epochs", 10))
    batches_per_epoch = math.ceil(len(records) / micro_batch)
    updates_per_epoch = math.ceil(batches_per_epoch / grad_accum)
    total_updates = max(1, epochs * updates_per_epoch)
    warmup_updates = round(total_updates * float(train_cfg.get("warmup_ratio", 0.05)))

    def lr_factor(step: int) -> float:
        if step < warmup_updates:
            return (step + 1) / max(1, warmup_updates)
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_epoch = 0
    global_step = 0
    if payload is not None:
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.snapshot.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_path = output_dir / "train.jsonl"
    negative_count = int(train_cfg.get("negative_count", 4095))
    strict_negatives = bool(train_cfg.get("strict_negative_count", True))
    local_min_chars = int(train_cfg.get("local_min_chars", 2))
    local_max_chars = int(train_cfg.get("local_max_chars", 8))
    use_bf16 = bool(train_cfg.get("bf16", True)) and str(device).startswith("cuda")
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    with log_path.open("a", encoding="utf-8") as log_stream:
        for epoch in range(start_epoch, epochs):
            model.train()
            model.set_epoch(epoch)
            shuffled = list(records)
            random.Random(seed + epoch).shuffle(shuffled)
            optimizer.zero_grad(set_to_none=True)
            running = {"loss": 0.0, "global": 0.0, "local": 0.0, "batches": 0}
            epoch_batches = list(batched(shuffled, micro_batch))
            for batch_index, batch_records in enumerate(epoch_batches):
                transcripts = [str(record["text"]) for record in batch_records]
                positives = [
                    deterministic_local_positive(
                        transcript,
                        utt_id=str(record["utt_id"]),
                        epoch=epoch,
                        seed=seed,
                        min_chars=local_min_chars,
                        max_chars=local_max_chars,
                    )
                    for record, transcript in zip(batch_records, transcripts)
                ]
                spoken_terms = batch_negative_exclusions(
                    transcripts,
                    min_chars=local_min_chars,
                    max_chars=local_max_chars,
                )
                excluded_negatives = spoken_terms | set(positives)
                negatives = sample_shared_negatives(
                    negative_vocabulary,
                    excluded_negatives,
                    negative_count,
                    seed=seed,
                    epoch=epoch,
                    step=batch_index,
                    strict=strict_negatives,
                )
                candidates = list(dict.fromkeys([*positives, *negatives]))
                qwen_features = [
                    _extract_qwen_features(
                        model,
                        processor,
                        read_wav_mono_float(record["audio"], 16000),
                    )
                    for record in batch_records
                ]
                with autocast():
                    audio_sequences = [model.encode_audio_features(item) for item in qwen_features]
                    audio_frames, frame_mask = _pad_audio(audio_sequences)
                    transcript_keys = _token_keys(model, processor, transcripts)
                    hotword_keys = _token_keys(model, processor, candidates)
                    global_mask = torch.as_tensor(
                        equality_positive_mask(transcripts, transcripts), device=device
                    )
                    local_mask = torch.as_tensor(
                        equality_positive_mask(positives, candidates), device=device
                    )
                    losses = glclap_loss(
                        audio_frames,
                        transcript_keys,
                        hotword_keys,
                        global_mask,
                        local_mask,
                        temperature=model.temperature,
                        frame_mask=frame_mask,
                        global_weight=float(loss_cfg.get("global_weight", 1.0)),
                        local_weight=float(loss_cfg.get("local_weight", 1.0)),
                    )
                    group_start = (batch_index // grad_accum) * grad_accum
                    actual_accum = min(grad_accum, len(epoch_batches) - group_start)
                    scaled_loss = losses.loss / actual_accum
                scaled_loss.backward()
                running["loss"] += float(losses.loss.detach())
                running["global"] += float(losses.global_loss.detach())
                running["local"] += float(losses.local_loss.detach())
                running["batches"] += 1
                should_step = (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(epoch_batches)
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for group in optimizer.param_groups for parameter in group["params"]],
                        float(train_cfg.get("grad_clip", 1.0)),
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            denominator = max(1, int(running["batches"]))
            event = {
                "epoch": epoch,
                "global_step": global_step,
                "loss": running["loss"] / denominator,
                "global_loss": running["global"] / denominator,
                "local_loss": running["local"] / denominator,
                "temperature": float(model.temperature.detach()),
                "mode": model.mode,
                "projector_trainable": bool(
                    model.retrieval_projector is not None
                    and any(p.requires_grad for p in model.retrieval_projector.parameters())
                ),
                "learning_rates": {group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)},
            }
            line = json.dumps(event, ensure_ascii=False, sort_keys=True)
            print(line, flush=True)
            log_stream.write(line + "\n")
            log_stream.flush()
            save_glclap_checkpoint(
                output_dir / f"epoch-{epoch:02d}.pt",
                model,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
            )
            save_glclap_checkpoint(
                output_dir / "last.pt",
                model,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
            )


if __name__ == "__main__":
    main()
