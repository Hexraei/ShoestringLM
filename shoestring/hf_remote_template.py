"""Remote-code companion for the ShoestringLM Hugging Face export.

The export directory carries a COPY of this file (with the full model
definition inlined at the marker below) plus config.json and
model.safetensors. transformers loads it when the export is opened with
trust_remote_code=True, so the model works with plain
AutoConfig/AutoModelForCausalLM - no ShoestringLM checkout required.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class ShoestringConfig(PretrainedConfig):
    """Config for the shoestring decoder-only transformer."""

    model_type = "shoestring"

    def __init__(
        self,
        name: str = "shoestring-30m",
        vocab_size: int = 8192,
        d_model: int = 384,
        n_layers: int = 20,
        n_heads: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.rope_theta = rope_theta
        defaults = {
            "tie_word_embeddings": False,  # head is tied by construction, not by key
            "max_position_embeddings": max_seq_len,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)


# The pasted model code only reads attributes off cfg, so the HF config
# object itself satisfies its ModelConfig type hint.
ModelConfig = ShoestringConfig

# __MODEL_SOURCE__


class ShoestringLMForCausalLM(PreTrainedModel):
    """PreTrainedModel wrapper: state dict keys live under `model.`."""

    config_class = ShoestringConfig
    base_model_prefix = "model"

    def __init__(self, config: ShoestringConfig):
        super().__init__(config)
        self.model = ShoestringLM(config)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Causal attention with no padding in eval batches: the mask is
        # accepted for API compatibility and intentionally unused.
        logits, loss = self.model(input_ids, labels)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def get_output_embeddings(self):
        # Tied head: logits are computed as x @ embed.weight.T inside the
        # model, so there is no separate output module to expose or re-tie.
        return None
