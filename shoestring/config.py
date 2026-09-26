"""Model configuration: a plain dataclass that round-trips to JSON."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    name: str = "shoestring-30m"
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 20
    n_heads: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls(**json.load(f))

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
