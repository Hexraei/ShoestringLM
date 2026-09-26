"""Generate text from a checkpoint."""
from __future__ import annotations

import argparse

import torch

from .config import ModelConfig
from .model import ShoestringLM
from .tokenizer import load_tokenizer, BOS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = ShoestringLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = load_tokenizer(ckpt["tokenizer"])
    ids = torch.tensor([[BOS] + tok.encode(args.prompt).ids])
    out = model.generate(ids, max_new=args.max_new, temperature=args.temperature)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
