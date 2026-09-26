"""Wrap the checkpoint as a tiny Hugging Face model so lm-eval can load it."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shoestring.config import ModelConfig
from shoestring.model import ShoestringLM

def main() -> None:
    ckpt_path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = ShoestringLM(cfg)
    model.load_state_dict(ckpt["model"])
    torch.save(model.state_dict(), f"{out}/pytorch_model.bin")
    cfg.to_json(f"{out}/shoestring_config.json")
    import shutil
    shutil.copy(ckpt["tokenizer"], f"{out}/tokenizer.json")
    print(f"exported to {out} - load with the ShoestringLM class for lm-eval runs")

if __name__ == "__main__":
    main()
