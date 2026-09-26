"""Training loop: AdamW, cosine schedule with warmup, gradient clipping.

Every run writes three things next to the checkpoint:
- log.csv: step, train loss, val loss, learning rate, tokens/sec
- training_report.json: hardware, wall time, token count, FLOP estimate
  (the hackathon requires hardware + time + compute in the README;
  this file is where those numbers come from)
- checkpoint.pt: model weights + config + tokenizer path
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time

import torch

from .config import ModelConfig
from .model import ShoestringLM
from .data import load_texts, pack_ids
from .tokenizer import train_tokenizer, load_tokenizer


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * 0.1 + 0.5 * peak * 0.9 * (1 + math.cos(math.pi * p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="roneneldan/TinyStories")
    ap.add_argument("--split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--out", default="runs/shoestring")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    cfg = ModelConfig.from_json(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok_path = f"{args.out}/tokenizer.json"
    print(f"[data] loading {args.data} ({args.split}) ...", flush=True)
    texts = load_texts(args.data, args.split, args.limit, args.text_field)
    print(f"[tokenizer] training BPE vocab={cfg.vocab_size} on {len(texts)} texts ...", flush=True)
    train_tokenizer(texts, cfg.vocab_size, tok_path)
    tok = load_tokenizer(tok_path)

    print("[data] packing sequences ...", flush=True)
    rows = pack_ids(texts, tok, cfg.max_seq_len)
    val_texts = load_texts(args.data, args.val_split, args.val_limit, args.text_field)
    val_rows = pack_ids(val_texts, tok, cfg.max_seq_len)
    print(f"[data] {len(rows)} train rows, {len(val_rows)} val rows", flush=True)

    model = ShoestringLM(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {total_params:,} trainable parameters", flush=True)
    assert total_params <= 50_000_000, "over the 50M cap"
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    g = torch.Generator().manual_seed(42)

    def batch(rows, bs):
        idx = torch.randint(len(rows), (bs,), generator=g)
        x = torch.tensor([rows[i][0] for i in idx]).to(device)
        y = torch.tensor([rows[i][1] for i in idx]).to(device)
        return x, y

    t0 = time.time()
    tokens_seen = 0
    with open(f"{args.out}/log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec"])
        for step in range(args.steps):
            lr = lr_at(step, args.steps, args.lr, args.warmup)
            for gp in opt.param_groups:
                gp["lr"] = lr
            x, y = batch(rows, args.batch)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tokens_seen += x.numel()
            if step % args.val_every == 0 or step == args.steps - 1:
                model.eval()
                with torch.no_grad():
                    vx, vy = batch(val_rows, min(args.batch, len(val_rows)))
                    _, vloss = model(vx, vy)
                model.train()
                el = max(time.time() - t0, 1e-9)
                tps = int(tokens_seen / el)
                print(f"step {step:>6} | train {loss.item():.4f} | val {vloss.item():.4f} | lr {lr:.2e} | {tps} tok/s", flush=True)
                w.writerow([step, f"{loss.item():.4f}", f"{vloss.item():.4f}", f"{lr:.2e}", tps])
                f.flush()

    wall = time.time() - t0
    torch.save({"model": model.state_dict(), "config": vars(cfg) if hasattr(cfg, "__dict__") else None,
                "tokenizer": tok_path}, f"{args.out}/checkpoint.pt")
    cfg.to_json(f"{args.out}/model_config.json")
    flops = 6 * total_params * tokens_seen
    report = {
        "hardware": platform.processor() or platform.machine(),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else f"CPU ({platform.machine()})",
        "wall_time_seconds": round(wall, 1),
        "tokens_trained": tokens_seen,
        "estimated_flops": f"{flops:.3e}",
        "steps": args.steps,
        "batch": args.batch,
        "seq_len": cfg.max_seq_len,
        "dataset": f"{args.data} ({args.split}, limit={args.limit})",
        "trainable_parameters": total_params,
    }
    with open(f"{args.out}/training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {wall/60:.1f} min, report + checkpoint in {args.out}/", flush=True)


if __name__ == "__main__":
    main()
