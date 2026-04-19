"""Run a loaded Transformer in a way that's useful for training.

Tinygrad's `Transformer.forward` is optimized for text *generation*:
    - it returns only the last position's logits
    - it applies Gumbel-max sampling internally (returns a sampled token ID)
    - it's wrapped in a TinyJit which specializes on exact shapes

For fine-tuning we need the opposite:
    - raw logits at every position (so we can compute next-token loss across
      the whole sequence)
    - no sampling (we want the probabilities themselves, not a sampled token)
    - no aggressive JIT (graph-capture interferes with autograd)

This module exposes `get_logits(...)` which does that.
"""
from __future__ import annotations

from tinygrad import Tensor
from tinygrad.llm.model import Transformer


def get_logits(model: Transformer, tokens: Tensor, start_pos: int = 0) -> Tensor:
    """Run a forward pass, return raw logits at every position.

    Args:
        model: a `Transformer` with weights loaded
        tokens: shape (B, T) int32 or int64 token IDs
        start_pos: for KV-cache-aware incremental decoding; 0 for a full
                   forward pass on a fresh batch (what training wants)

    Returns:
        logits of shape (B, T, vocab_size), dtype float32

    Why this isn't just `model(tokens, 0, 1.0)`:
        - `Transformer.__call__` dispatches into a TinyJit-wrapped `forward`
        - `forward` slices `[:, -1, :]` (last-token only) and then applies
          Gumbel-max sampling, returning token IDs, not logits
        - For training we need logits at every position to compute
          cross-entropy over the full sequence
    """
    # Embed tokens → (B, T, D)
    x = model.token_embd(tokens).float()

    # Apply each transformer block sequentially. `start_pos` is threaded
    # through to handle KV-cache positioning; 0 means "start from scratch".
    for block in model.blk:
        x = block(x, start_pos)

    # Final layer norm, then project to vocabulary → (B, T, V)
    logits = model.output(model.output_norm(x))
    return logits
