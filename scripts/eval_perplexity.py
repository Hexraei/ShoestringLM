"""Perplexity on a held-out slice of WikiText-103 (GIBC eval requirement)."""
from __future__ import annotations

import argparse
import math

import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shoestring.config import ModelConfig
from shoestring.model import ShoestringLM
from shoestring.tokenizer import load_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--rows", type=int, default=200)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=f"validation[:{args.rows}]")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ShoestringLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = load_tokenizer(ckpt["tokenizer"])

    nll, ntok = 0.0, 0
    with torch.no_grad():
        for row in ds:
            text = row["text"].strip()
            if len(text) < 40:
                continue
            ids = tok.encode(text).ids[: cfg.max_seq_len]
            if len(ids) < 8:
                continue
            x = torch.tensor([ids[:-1]]).to(device)
            y = torch.tensor([ids[1:]]).to(device)
            _, loss = model(x, y)
            nll += loss.item() * y.numel()
            ntok += y.numel()
    ppl = math.exp(nll / max(ntok, 1))
    print(f"WikiText-103 held-out slice ({args.rows} rows requested): perplexity = {ppl:.2f} over {ntok:,} tokens")


if __name__ == "__main__":
    main()
