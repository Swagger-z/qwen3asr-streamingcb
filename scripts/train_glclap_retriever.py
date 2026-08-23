"""Train GLCLAP audio/text adapters on frozen Qwen3-ASR-0.6B features."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from asr.audio_io import read_wav_mono_array
from asr.config import load_config, require_mapping
from asr.data.manifest import manifest_key, manifest_source, manifest_target, validate_manifest_schema
from asr.contextual.glclap_cache import FrozenTextEmbeddingCache, QwenFeatureCache
from asr.contextual.glclap_data import (
    batch_negative_exclusions,
    deterministic_local_positive,
    equality_positive_mask,
    SharedNegativeSampler,
)
from asr.contextual.glclap_distributed import (
    resolve_gradient_accumulation,
    shard_epoch_records,
    shard_evaluation_records,
)
from asr.contextual.glclap_loss import glclap_loss
from asr.contextual.glclap_model import save_glclap_checkpoint
from asr.contextual.glclap_validation import ValidationSchedule, retrieval_rank_metrics
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
    parser.add_argument("--manifest", required=True, help="AISHELL-1 JSONL: key/source/target")
    parser.add_argument("--dev-manifest", required=True, help="held-out JSONL: key/source/target")
    parser.add_argument("--negative-catalog", required=True, help="JSONL, one term per line, or TERM COUNT text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-cache-dir",
        help="shared disk cache for frozen Qwen audio features; strongly recommended",
    )
    parser.add_argument("--resume")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _extract_qwen_features_batch(
    model: Any,
    processor: Any,
    waveforms: Sequence[np.ndarray],
    *,
    packed: bool,
) -> list[Any]:
    """Batch feature extraction and invoke the frozen Qwen tower once when packed."""

    if not waveforms:
        return []
    batch = processor.feature_extractor(
        [np.asarray(waveform, dtype=np.float32) for waveform in waveforms],
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    device = next(model.adapters.parameters()).device
    qwen_dtype = next(model.encoder.audio_tower.parameters()).dtype
    padded = batch["input_features"].to(device=device, dtype=qwen_dtype, non_blocking=True)
    feature_lens = batch["attention_mask"].sum(dim=1).long().to(device, non_blocking=True)
    features = [padded[index] for index in range(padded.shape[0])]
    return model.encoder.extract_audio_features_batch(features, feature_lens, packed=packed)


def _load_waveforms(
    records: Sequence[Mapping[str, Any]],
    executor: Executor | None,
) -> list[np.ndarray]:
    """Load a batch of WAVs, optionally using persistent worker threads."""

    sources = [manifest_source(record) for record in records]
    if executor is None:
        return [read_wav_mono_array(source, 16000) for source in sources]
    return list(executor.map(read_wav_mono_array, sources))


def _audio_features_for_records(
    model: Any,
    processor: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    cache: QwenFeatureCache | None,
    executor: Executor | None,
    packed: bool,
) -> tuple[list[Any], dict[str, float]]:
    """Load cached features and batch-encode only cache misses."""

    started = time.perf_counter()
    outputs: list[Any | None] = [None] * len(records)
    missing_indices: list[int] = []
    for index, record in enumerate(records):
        cached = (
            cache.get(manifest_key(record), manifest_source(record))
            if cache is not None
            else None
        )
        if cached is None:
            missing_indices.append(index)
        else:
            outputs[index] = cached
    cache_load_seconds = time.perf_counter() - started

    io_started = time.perf_counter()
    missing_records = [records[index] for index in missing_indices]
    waveforms = _load_waveforms(missing_records, executor)
    audio_io_seconds = time.perf_counter() - io_started

    encoder_started = time.perf_counter()
    encoded = _extract_qwen_features_batch(model, processor, waveforms, packed=packed)
    encoder_submit_seconds = time.perf_counter() - encoder_started

    write_started = time.perf_counter()
    cached_parts: Sequence[Any] | None = None
    if cache is not None and encoded:
        import torch

        selected = [
            item.post_projector
            if cache.feature_kind == "post_projector"
            else item.pre_projector
            for item in encoded
        ]
        selected_lengths = [int(item.shape[0]) for item in selected]
        cached_parts = torch.cat(selected, dim=0).detach().to(device="cpu").split(
            selected_lengths, dim=0
        )
    for encoded_index, (index, features) in enumerate(zip(missing_indices, encoded)):
        record = records[index]
        outputs[index] = features
        if cache is not None and cached_parts is not None:
            cache.put_tensor(
                manifest_key(record),
                manifest_source(record),
                cached_parts[encoded_index],
            )
    cache_write_seconds = time.perf_counter() - write_started
    if any(value is None for value in outputs):
        raise RuntimeError("audio feature batch was not completely populated")
    return list(outputs), {
        "audio_cache_load_seconds": cache_load_seconds,
        "audio_io_seconds": audio_io_seconds,
        "audio_encoder_submit_seconds": encoder_submit_seconds,
        "audio_cache_write_seconds": cache_write_seconds,
    }


def _pad_audio(frames: list[Any]) -> tuple[Any, Any]:
    import torch

    if not frames:
        raise ValueError("cannot pad an empty audio batch")
    lengths = [item.shape[0] for item in frames]
    padded = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)
    positions = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0)
    mask = positions < torch.tensor(lengths, device=padded.device).unsqueeze(1)
    return padded, mask


def _evaluate(
    model: Any,
    processor: Any,
    records: Sequence[Mapping[str, Any]],
    negative_sampler: SharedNegativeSampler,
    *,
    batch_size: int,
    negative_count: int,
    strict_negatives: bool,
    local_min_chars: int,
    local_max_chars: int,
    seed: int,
    loss_cfg: Mapping[str, Any],
    device: Any,
    autocast: Any,
    audio_cache: QwenFeatureCache | None,
    text_cache: FrozenTextEmbeddingCache,
    audio_executor: Executor | None,
    packed_audio: bool,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, float | int]:
    """Evaluate a fixed held-out set with loss and local retrieval ranks."""

    import torch

    model.eval()
    totals = {
        "loss": 0.0,
        "global_loss": 0.0,
        "local_loss": 0.0,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "recall_at_20": 0.0,
        "recall_at_50": 0.0,
        "mrr": 0.0,
    }
    seen = 0
    num_batches = 0
    with torch.inference_mode():
        for batch_index, batch_records in enumerate(batched(records, batch_size)):
            transcripts = [manifest_target(record) for record in batch_records]
            positives = [
                deterministic_local_positive(
                    transcript,
                    utt_id=manifest_key(record),
                    epoch=0,
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
            negatives = negative_sampler.sample(
                spoken_terms | set(positives),
                negative_count,
                seed=seed,
                epoch=0,
                step=batch_index * world_size + rank,
                strict=strict_negatives,
            )
            candidates = list(dict.fromkeys([*positives, *negatives]))
            qwen_features, _batch_timings = _audio_features_for_records(
                model,
                processor,
                batch_records,
                cache=audio_cache,
                executor=audio_executor,
                packed=packed_audio,
            )
            with autocast():
                audio_sequences = [model.encode_audio_features(item) for item in qwen_features]
                audio_frames, frame_mask = _pad_audio(audio_sequences)
                transcript_keys = model.encode_text_embeddings(
                    text_cache.get(model, processor, transcripts)
                )
                hotword_keys = model.encode_text_embeddings(
                    text_cache.get(model, processor, candidates)
                )
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
            count = len(batch_records)
            rank_metrics = retrieval_rank_metrics(
                losses.local_logits.detach().float().cpu().numpy(),
                local_mask.detach().cpu().numpy(),
            )
            totals["loss"] += float(losses.loss.detach()) * count
            totals["global_loss"] += float(losses.global_loss.detach()) * count
            totals["local_loss"] += float(losses.local_loss.detach()) * count
            for name in (
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "recall_at_20",
                "recall_at_50",
                "mrr",
            ):
                totals[name] += rank_metrics[name] * count
            seen += count
            num_batches += 1
    if seen == 0:
        result = {name: 0.0 for name in totals}
        result["num_examples"] = 0
        result["num_batches"] = 0
        return result
    result: dict[str, float | int] = {name: value / seen for name, value in totals.items()}
    result["num_examples"] = seen
    result["num_batches"] = num_batches
    return result


def _merge_distributed_validation(metrics: Mapping[str, float | int], device: Any) -> dict[str, float | int]:
    """Aggregate example-weighted validation metrics across all ranks."""

    import torch

    names = (
        "loss",
        "global_loss",
        "local_loss",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "mrr",
    )
    count = int(metrics["num_examples"])
    tensor = torch.tensor(
        [
            *[float(metrics[name]) * count for name in names],
            float(count),
            float(metrics["num_batches"]),
        ],
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    reduced = tensor.cpu().tolist()
    total_count = int(reduced[-2])
    if total_count <= 0:
        raise ValueError("distributed validation manifest is empty")
    result: dict[str, float | int] = {
        name: reduced[index] / total_count for index, name in enumerate(names)
    }
    result["num_examples"] = total_count
    result["num_batches"] = int(reduced[-1])
    return result


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
    eval_cfg = dict(require_mapping(config, "evaluation"))
    validation_schedule = ValidationSchedule.from_config(eval_cfg)
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    distributed_cfg = dict(config.get("distributed", {}))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid torchrun WORLD_SIZE/RANK environment")
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-GPU GLCLAP training requires CUDA")
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is invalid for {torch.cuda.device_count()} visible GPUs"
            )
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend=str(distributed_cfg.get("backend", "nccl")),
            init_method="env://",
        )
    is_main_process = rank == 0
    runtime_config = copy.deepcopy(config)
    requested_device = str(runtime_config.get("runtime", {}).get("device", "cuda"))
    if distributed and requested_device.startswith("cuda"):
        runtime_config.setdefault("runtime", {})["device"] = f"cuda:{local_rank}"

    records = jsonl_records(args.manifest)
    if not records:
        raise ValueError("training manifest is empty")
    validate_manifest_schema(records)
    dev_records = jsonl_records(args.dev_manifest)
    if not dev_records:
        raise ValueError("validation manifest is empty")
    validate_manifest_schema(dev_records)
    overlapping_keys = {manifest_key(record) for record in records} & {
        manifest_key(record) for record in dev_records
    }
    if overlapping_keys:
        examples = ", ".join(sorted(overlapping_keys)[:5])
        raise ValueError(f"train/dev manifests overlap on {len(overlapping_keys)} keys: {examples}")
    eval_seed = int(eval_cfg.get("seed", seed))
    eval_max_samples = int(eval_cfg.get("max_samples", 0))
    if eval_max_samples < 0:
        raise ValueError("evaluation.max_samples must be non-negative")
    if eval_max_samples:
        fixed_dev_records = list(dev_records)
        random.Random(eval_seed).shuffle(fixed_dev_records)
        dev_records = fixed_dev_records[:eval_max_samples]
    negative_vocabulary = load_negative_vocabulary(args.negative_catalog)
    negative_sampler = SharedNegativeSampler(negative_vocabulary)
    model, _runtime, processor, payload = build_glclap_runtime(
        runtime_config, checkpoint=args.resume
    )
    device = next(model.adapters.parameters()).device
    training_model: Any = model
    if distributed:
        if model.retrieval_projector is not None:
            for parameter in model.retrieval_projector.parameters():
                parameter.requires_grad_(True)
        training_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    adapter_lr = float(train_cfg.get("adapter_lr", 3e-4))
    projector_lr = float(train_cfg.get("projector_lr", 3e-5))
    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(adapter_lr, projector_lr),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    micro_batch = int(train_cfg.get("micro_batch_size", 8))
    global_batch_size = int(train_cfg.get("global_batch_size", 0))
    if global_batch_size > 0:
        grad_accum = resolve_gradient_accumulation(
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch,
            world_size=world_size,
        )
    else:
        grad_accum = int(train_cfg.get("gradient_accumulation_steps", 48))
        global_batch_size = micro_batch * grad_accum * world_size
    epochs = int(train_cfg.get("epochs", 10))
    records_per_rank = math.ceil(len(records) / world_size)
    batches_per_epoch = math.ceil(records_per_rank / micro_batch)
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
    best_metric_name = str(eval_cfg.get("selection_metric", "recall_at_50"))
    best_metric_value = float("-inf")
    best_epoch = -1
    best_global_step = 0
    if payload is not None:
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        restored_state = payload.get("training_state") or {}
        best_metric_name = str(restored_state.get("best_metric_name", best_metric_name))
        best_metric_value = float(restored_state.get("best_metric_value", best_metric_value))
        best_epoch = int(restored_state.get("best_epoch", best_epoch))
        best_global_step = int(restored_state.get("best_global_step", best_global_step))

    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.snapshot.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if distributed:
        torch.distributed.barrier()
    log_path = output_dir / "train.jsonl"
    negative_count = int(train_cfg.get("negative_count", 4095))
    strict_negatives = bool(train_cfg.get("strict_negative_count", True))
    local_min_chars = int(train_cfg.get("local_min_chars", 2))
    local_max_chars = int(train_cfg.get("local_max_chars", 8))
    train_local_enabled = float(loss_cfg.get("local_weight", 1.0)) != 0.0
    use_bf16 = bool(train_cfg.get("bf16", True)) and str(device).startswith("cuda")
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
    eval_batch_size = int(eval_cfg.get("batch_size", micro_batch))
    eval_negative_count = int(eval_cfg.get("negative_count", negative_count))
    eval_strict_negatives = bool(eval_cfg.get("strict_negative_count", strict_negatives))
    audio_batching = str(train_cfg.get("audio_batching", "packed")).strip().lower()
    if audio_batching not in {"packed", "serial"}:
        raise ValueError("training.audio_batching must be packed or serial")
    packed_audio = audio_batching == "packed"
    audio_loader_workers = int(train_cfg.get("audio_loader_workers", 4))
    if audio_loader_workers < 0:
        raise ValueError("training.audio_loader_workers must be non-negative")
    audio_executor: Executor | None = (
        ThreadPoolExecutor(max_workers=audio_loader_workers, thread_name_prefix="glclap-wav")
        if audio_loader_workers > 0
        else None
    )
    text_cache = FrozenTextEmbeddingCache(
        max_entries=int(train_cfg.get("text_cache_max_entries", 20000)),
        encode_batch_size=int(train_cfg.get("text_cache_batch_size", 1024)),
    )
    audio_cache: QwenFeatureCache | None = None
    if args.feature_cache_dir:
        model_cfg = dict(runtime_config.get("model", {}))
        namespace = (
            f"{model_cfg.get('qwen_model', 'unknown')}|qwen-asr-0.0.6"
            f"|audio_batching={audio_batching}"
        )
        feature_kind = (
            "post_projector"
            if model.retrieval_projector is None
            else "pre_projector"
        )
        audio_cache = QwenFeatureCache(
            args.feature_cache_dir,
            namespace=namespace,
            feature_kind=feature_kind,
            device=device,
            dtype=next(model.encoder.audio_tower.parameters()).dtype,
        )
    if bool(train_cfg.get("prewarm_text_cache", True)):
        text_cache.get(model, processor, negative_vocabulary)
    if eval_batch_size <= 0:
        raise ValueError("evaluation.batch_size must be positive")
    last_eval_global_step = -1

    def checkpoint_training_state() -> dict[str, Any]:
        return {
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "best_epoch": best_epoch,
            "best_global_step": best_global_step,
            "world_size": world_size,
            "global_batch_size": global_batch_size,
            "gradient_accumulation_steps": grad_accum,
        }

    def run_validation(epoch: int, trigger: str, log_stream: Any) -> None:
        nonlocal best_metric_value, best_epoch, best_global_step, last_eval_global_step
        local_dev_records = shard_evaluation_records(
            dev_records, world_size=world_size, rank=rank
        )
        metrics = _evaluate(
            model,
            processor,
            local_dev_records,
            negative_sampler,
            batch_size=eval_batch_size,
            negative_count=eval_negative_count,
            strict_negatives=eval_strict_negatives,
            local_min_chars=local_min_chars,
            local_max_chars=local_max_chars,
            seed=eval_seed,
            loss_cfg=loss_cfg,
            device=device,
            autocast=autocast,
            audio_cache=audio_cache,
            text_cache=text_cache,
            audio_executor=audio_executor,
            packed_audio=packed_audio,
            rank=rank,
            world_size=world_size,
        )
        if distributed:
            metrics = _merge_distributed_validation(metrics, device)
        if best_metric_name not in metrics:
            raise ValueError(f"unknown evaluation.selection_metric: {best_metric_name}")
        current_metric = float(metrics[best_metric_name])
        improved = current_metric > best_metric_value
        if improved:
            best_metric_value = current_metric
            best_epoch = epoch
            best_global_step = global_step
        if is_main_process:
            event = {
                "event": "validation",
                "trigger": trigger,
                "epoch": epoch,
                "global_step": global_step,
                "world_size": world_size,
                **metrics,
            }
            line = json.dumps(event, ensure_ascii=False, sort_keys=True)
            print(line, flush=True)
            log_stream.write(line + "\n")
            log_stream.flush()
            if improved:
                save_glclap_checkpoint(
                    output_dir / "best.pt",
                    model,
                    config=config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    training_state=checkpoint_training_state(),
                )
        last_eval_global_step = global_step
        training_model.train()
        if not distributed:
            model.set_epoch(epoch)

    log_context = log_path.open("a", encoding="utf-8") if is_main_process else nullcontext(None)
    with log_context as log_stream:
        for epoch in range(start_epoch, epochs):
            epoch_started = time.perf_counter()
            training_model.train()
            if not distributed:
                model.set_epoch(epoch)
            shuffled = list(records)
            random.Random(seed + epoch).shuffle(shuffled)
            optimizer.zero_grad(set_to_none=True)
            running = torch.zeros(4, dtype=torch.float64, device=device)
            performance = {
                "samples": 0.0,
                "audio_cache_load_seconds": 0.0,
                "audio_io_seconds": 0.0,
                "audio_encoder_submit_seconds": 0.0,
                "audio_cache_write_seconds": 0.0,
                "text_prepare_seconds": 0.0,
            }
            rank_records = shard_epoch_records(shuffled, world_size=world_size, rank=rank)
            epoch_batches = list(batched(rank_records, micro_batch))
            for batch_index, batch_records in enumerate(epoch_batches):
                transcripts = [manifest_target(record) for record in batch_records]
                if train_local_enabled:
                    positives = [
                        deterministic_local_positive(
                            transcript,
                            utt_id=manifest_key(record),
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
                    negatives = negative_sampler.sample(
                        spoken_terms | set(positives),
                        negative_count,
                        seed=seed,
                        epoch=epoch,
                        step=batch_index * world_size + rank,
                        strict=strict_negatives,
                    )
                    candidates = list(dict.fromkeys([*positives, *negatives]))
                else:
                    positives = [""] * len(batch_records)
                    candidates = []
                qwen_features, audio_timings = _audio_features_for_records(
                    model,
                    processor,
                    batch_records,
                    cache=audio_cache,
                    executor=audio_executor,
                    packed=packed_audio,
                )
                for name, value in audio_timings.items():
                    performance[name] += value
                text_started = time.perf_counter()
                transcript_embeddings = text_cache.get(model, processor, transcripts)
                hotword_embeddings = text_cache.get(model, processor, candidates)
                performance["text_prepare_seconds"] += time.perf_counter() - text_started
                performance["samples"] += len(batch_records)
                should_step = (
                    (batch_index + 1) % grad_accum == 0
                    or batch_index + 1 == len(epoch_batches)
                )
                sync_context = (
                    training_model.no_sync() if distributed and not should_step else nullcontext()
                )
                with sync_context:
                    with autocast():
                        audio_sequences, transcript_keys, hotword_keys, temperature = training_model(
                            qwen_features,
                            None,
                            None,
                            None,
                            None,
                            transcript_embeddings=transcript_embeddings,
                            hotword_embeddings=hotword_embeddings,
                        )
                        audio_frames, frame_mask = _pad_audio(audio_sequences)
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
                            temperature=temperature,
                            frame_mask=frame_mask,
                            global_weight=float(loss_cfg.get("global_weight", 1.0)),
                            local_weight=float(loss_cfg.get("local_weight", 1.0)),
                            compute_local=train_local_enabled,
                        )
                        group_start = (batch_index // grad_accum) * grad_accum
                        actual_accum = min(grad_accum, len(epoch_batches) - group_start)
                        scaled_loss = losses.loss / actual_accum
                    scaled_loss.backward()
                running[0] += losses.loss.detach().double()
                running[1] += losses.global_loss.detach().double()
                running[2] += losses.local_loss.detach().double()
                running[3] += 1
                if should_step:
                    if distributed and model.retrieval_projector is not None and epoch < 1:
                        # DDP must register these parameters at construction;
                        # discarding epoch-0 gradients preserves the warm-start freeze.
                        for parameter in model.retrieval_projector.parameters():
                            parameter.grad = None
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for group in optimizer.param_groups for parameter in group["params"]],
                        float(train_cfg.get("grad_clip", 1.0)),
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if validation_schedule.after_optimizer_step(global_step):
                        run_validation(epoch, "steps", log_stream)

            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_started
            if distributed:
                torch.distributed.all_reduce(running, op=torch.distributed.ReduceOp.SUM)
                perf_names = tuple(performance)
                perf_tensor = torch.tensor(
                    [performance[name] for name in perf_names],
                    dtype=torch.float64,
                    device=device,
                )
                torch.distributed.all_reduce(perf_tensor, op=torch.distributed.ReduceOp.SUM)
                performance = dict(zip(perf_names, perf_tensor.cpu().tolist()))
                elapsed_tensor = torch.tensor(epoch_seconds, dtype=torch.float64, device=device)
                torch.distributed.all_reduce(elapsed_tensor, op=torch.distributed.ReduceOp.MAX)
                epoch_seconds = float(elapsed_tensor.cpu())
            running_values = running.cpu().tolist()
            denominator = max(1, int(running_values[3]))
            if is_main_process:
                event = {
                    "event": "train_epoch",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": running_values[0] / denominator,
                    "global_loss": running_values[1] / denominator,
                    "local_loss": running_values[2] / denominator,
                    "temperature": float(model.temperature.detach()),
                    "mode": model.mode,
                    "world_size": world_size,
                    "micro_batch_size": micro_batch,
                    "gradient_accumulation_steps": grad_accum,
                    "global_batch_size": global_batch_size,
                    "audio_batching": audio_batching,
                    "epoch_seconds": epoch_seconds,
                    "samples_per_second": performance["samples"] / max(epoch_seconds, 1e-9),
                    "performance_seconds": {
                        name: value / world_size
                        for name, value in performance.items()
                        if name != "samples"
                    },
                    "audio_feature_cache": audio_cache.stats() if audio_cache else None,
                    "text_embedding_cache": text_cache.stats(),
                    "projector_trainable": bool(
                        model.retrieval_projector is not None and epoch >= 1
                    ),
                    "learning_rates": {
                        group.get("name", str(i)): group["lr"]
                        for i, group in enumerate(optimizer.param_groups)
                    },
                }
                line = json.dumps(event, ensure_ascii=False, sort_keys=True)
                print(line, flush=True)
                log_stream.write(line + "\n")
                log_stream.flush()
            if validation_schedule.after_epoch():
                run_validation(epoch, "epoch", log_stream)
            elif epoch + 1 == epochs and last_eval_global_step != global_step:
                run_validation(epoch, "final", log_stream)
            if is_main_process:
                for checkpoint_path in (output_dir / f"epoch-{epoch:02d}.pt", output_dir / "last.pt"):
                    save_glclap_checkpoint(
                        checkpoint_path,
                        model,
                        config=config,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        training_state=checkpoint_training_state(),
                    )
            if distributed:
                torch.distributed.barrier()

    if audio_executor is not None:
        audio_executor.shutdown(wait=True)
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
