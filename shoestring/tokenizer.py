"""Train a byte-level BPE tokenizer on the training text.

Small vocab on purpose: 8192 tokens keeps the embedding matrix cheap
(it is tied to the output head, so every token costs two uses of one matrix).
"""
from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

SPECIALS = ["<pad>", "<bos>", "<eos>"]
PAD, BOS, EOS = 0, 1, 2


def train_tokenizer(texts, vocab_size: int, out_path: str) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, show_progress=False)
    tok.train_from_iterator(texts, trainer)
    tok.save(out_path)
    return tok


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def encode(tok: Tokenizer, text: str, add_special: bool = True) -> list[int]:
    ids = tok.encode(text).ids
    return ([BOS] + ids + [EOS]) if add_special else ids
