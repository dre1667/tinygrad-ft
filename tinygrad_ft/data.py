"""Tokenize and batch text examples for next-token prediction training.

For the simplest training case (pretraining / instruction-tuning with a
uniform format), we take a list of `{"text": "..."}` records, tokenize
each one, pad to the longest in the batch, and return two parallel
tensors: `input_ids` and `attention_mask`.

`attention_mask` here is a "loss mask" — positions where it's 0 get
`ignore_index=-1` during cross-entropy so padding doesn't contribute to
the loss. The transformer itself still attends to padding positions;
proper attention masking at the attention layer is a finer-grained
optimization we can add later.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypedDict

import numpy as np

from tinygrad import Tensor, dtypes

from .tokenizer import HFTokenizer


class TokenizedBatch(TypedDict):
    input_ids: Tensor          # (B, T) int32, the raw tokens
    loss_mask: Tensor          # (B, T) int32, 1 where real token, 0 where padded


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a .jsonl file into a list of dicts (one per line)."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tokenize_batch(
    examples: Iterable[dict],
    tokenizer: HFTokenizer,
    max_length: int = 128,
    pad_token_id: int = 0,
) -> TokenizedBatch:
    """Turn a list of {"text": "..."} examples into a padded tensor batch.

    Args:
        examples:     iterable of dicts, each containing a "text" field
        tokenizer:    an HFTokenizer instance
        max_length:   truncation cap; examples longer than this are clipped
        pad_token_id: padding ID (0 is safe for Qwen/Llama tokenizers)

    Returns:
        dict with:
            input_ids:   (B, T) int32, tokens (padded at right with pad_token_id)
            loss_mask:   (B, T) int32, 1 on real tokens, 0 on pads
    """
    id_lists: list[list[int]] = []
    for ex in examples:
        if "text" not in ex:
            raise KeyError(f"example missing 'text' field: {ex}")
        ids = tokenizer.encode(ex["text"])[:max_length]
        id_lists.append(ids)

    batch_len = max(len(ids) for ids in id_lists)
    B = len(id_lists)

    padded  = np.full((B, batch_len), pad_token_id, dtype=np.int32)
    mask    = np.zeros((B, batch_len),               dtype=np.int32)
    for i, ids in enumerate(id_lists):
        padded[i, :len(ids)] = ids
        mask[i,   :len(ids)] = 1

    return {
        "input_ids": Tensor(padded,  dtype=dtypes.int32, requires_grad=False),
        "loss_mask": Tensor(mask,    dtype=dtypes.int32, requires_grad=False),
    }
