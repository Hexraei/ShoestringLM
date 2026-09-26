"""Data: TinyStories by default, any HF text dataset or local .txt works.

TinyStories (Ronen Eldan & Yuanzhi Li, 2023) is a public dataset of short
synthetic stories. Small models learn real grammar from it fast, which is
exactly what a shoestring budget needs.
"""
from __future__ import annotations

import itertools


def load_texts(source: str, split: str, limit: int | None = None, text_field: str = "text"):
    if source.endswith(".txt"):
        with open(source) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset(source, split=split)
        texts = [r[text_field] for r in ds]
    if limit:
        texts = list(itertools.islice(texts, limit))
    return texts


def pack_ids(texts, tok, seq_len: int):
    """Tokenize everything and slice into fixed-length training rows."""
    from .tokenizer import encode
    stream: list[int] = []
    for t in texts:
        stream.extend(encode(tok, t))
    rows = []
    for i in range(0, len(stream) - seq_len - 1, seq_len):
        rows.append((stream[i : i + seq_len], stream[i + 1 : i + seq_len + 1]))
    return rows
