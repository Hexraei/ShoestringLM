"""Print the trainable parameter count with a per-part breakdown.

The GIBC rules count everything trainable, embeddings and output head
included. Our head is tied to the embedding matrix, so it appears once.
"""
from __future__ import annotations

import argparse
import json

from .config import ModelConfig
from .model import ShoestringLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = ModelConfig.from_json(args.config)
    model = ShoestringLM(cfg)
    parts = model.count_parameters()
    print(f"Model: {cfg.name}")
    for k, v in parts.items():
        print(f"  {k:<38} {v:>12,}")
    total = parts["TOTAL trainable parameters"]
    cap = 50_000_000
    print(f"\nCap check: {total:,} <= {cap:,} -> {'PASS' if total <= cap else 'FAIL'}")
    print(json.dumps(parts))


if __name__ == "__main__":
    main()
