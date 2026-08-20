"""Train the LayerNorm+Linear phoneme CTC head on frozen AuT features."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from asr.config import load_config, require_mapping


def _records(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    """Optimize only the phoneme head and write a reproducible checkpoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required; install the 'probe' extra") from exc
    from asr.contextual.probes import PhonemeCTCHead

    config = load_config(args.config, args.override)
    probe = dict(require_mapping(config, "probe"))
    seed = int(probe.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    records = _records(args.manifest)
    if not records:
        raise ValueError("empty feature manifest")
    head = PhonemeCTCHead(int(probe["encoder_dim"]), int(probe["vocab_size"]))
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(probe.get("learning_rate", 1e-3)))
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
    blank_id = int(probe.get("blank_id", 0))
    criterion = torch.nn.CTCLoss(blank=blank_id, zero_infinity=True)
    epochs = int(probe.get("epochs", 10))
    head.train()
    for epoch in range(start_epoch, epochs):
        random.shuffle(records)
        total_loss = 0.0
        for record in records:
            item = torch.load(record["features"], map_location="cpu")
            features = item["features"] if isinstance(item, dict) else item
            if features.ndim == 2:
                features = features.unsqueeze(0)
            targets = torch.tensor(record.get("target_ids", item.get("target_ids")), dtype=torch.long)
            logits = head(features.float())
            log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
            input_lengths = torch.tensor([logits.shape[1]], dtype=torch.long)
            target_lengths = torch.tensor([targets.numel()], dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(probe.get("grad_clip", 5.0)))
            optimizer.step()
            total_loss += float(loss.detach())
        print(json.dumps({"epoch": epoch, "loss": total_loss / len(records)}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs - 1,
            "config": config,
            "symbols": probe.get("symbols", []),
        },
        output,
    )


if __name__ == "__main__":
    main()
