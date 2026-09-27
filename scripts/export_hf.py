"""Wrap the checkpoint as a Hugging Face model so lm-eval (or anyone) can load it.

Writes a trust_remote_code export:
  config.json            - model_type "shoestring" + auto_map
  modeling_shoestring.py - self-contained remote code (model definition inlined)
  model.safetensors      - weights under the `model.` prefix
  tokenizer.json         - the trained BPE tokenizer
  tokenizer_config.json  - special tokens + max length for AutoTokenizer
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shoestring.config import ModelConfig
from shoestring.model import ShoestringLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(REPO_ROOT, "shoestring", "hf_remote_template.py")
MODEL_SRC = os.path.join(REPO_ROOT, "shoestring", "model.py")
MARKER = "# __MODEL_SOURCE__"
# Lines that are only valid inside the package, not in a standalone module.
DROP_LINES = {"from __future__ import annotations", "from .config import ModelConfig"}


def build_remote_module() -> str:
    with open(TEMPLATE) as f:
        template = f.read()
    with open(MODEL_SRC) as f:
        kept = [ln for ln in f.read().splitlines() if ln.strip() not in DROP_LINES]
    assert MARKER in template, "template marker missing"
    return template.replace(MARKER, "\n".join(kept))


def main() -> None:
    ckpt_path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = ShoestringLM(cfg)
    model.load_state_dict(ckpt["model"])

    from safetensors.torch import save_file
    prefixed = {f"model.{k}": v.contiguous() for k, v in model.state_dict().items()}
    save_file(prefixed, f"{out}/model.safetensors")

    config = {
        **vars(cfg),
        "model_type": "shoestring",
        "architectures": ["ShoestringLMForCausalLM"],
        "auto_map": {
            "AutoConfig": "modeling_shoestring.ShoestringConfig",
            "AutoModelForCausalLM": "modeling_shoestring.ShoestringLMForCausalLM",
        },
        "tie_word_embeddings": False,
        "max_position_embeddings": cfg.max_seq_len,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    with open(f"{out}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(f"{out}/modeling_shoestring.py", "w") as f:
        f.write(build_remote_module())

    import shutil
    tok_src = ckpt["tokenizer"]
    if not os.path.exists(tok_src):
        # checkpoints store the path relative to the training cwd;
        # fall back to resolving it next to the checkpoint itself
        tok_src = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), os.path.basename(tok_src))
    shutil.copy(tok_src, f"{out}/tokenizer.json")
    with open(f"{out}/tokenizer_config.json", "w") as f:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": cfg.max_seq_len,
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            },
            f,
            indent=2,
        )
    print(f"exported to {out} - load with AutoModelForCausalLM.from_pretrained('{out}', trust_remote_code=True)")


if __name__ == "__main__":
    main()
