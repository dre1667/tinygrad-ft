"""Run a loaded Transformer in a way that's useful for training.

Tinygrad's `Transformer.forward` is optimized for text *generation*:
    - it returns only the last position's logits
    - it applies Gumbel-max sampling internally (returns a sampled token ID)
    - it uses an in-place KV cache (STORE op) that tinygrad's autograd
      cannot differentiate through
    - it's wrapped in a TinyJit which specializes on exact shapes

For fine-tuning we need the opposite:
    - raw logits at every position (so we can compute next-token loss)
    - no sampling (we want the probabilities, not a sampled token)
    - no cache mutation (training processes full sequences fresh each step)
    - no aggressive JIT (graph-capture interferes with autograd)

This module exposes two functions:
    - `get_logits(model, tokens)`          — inference-safe forward pass
    - `get_logits_train(model, tokens)`    — training-safe (cache-free)
"""
from __future__ import annotations

import types

from tinygrad import Tensor
from tinygrad.llm.model import Transformer, apply_rope, precompute_freqs_cis


# ---------------------------------------------------------------------------
# Inference-mode forward pass (uses KV cache, suitable for generation)
# ---------------------------------------------------------------------------

def get_logits(model: Transformer, tokens: Tensor, start_pos: int = 0) -> Tensor:
    """Full-sequence logits using tinygrad's stock attention (KV cache enabled).

    Suitable for: forward-pass sanity checks, generation, parity testing.
    Not suitable for: training — the cache's STORE op breaks autograd.
    """
    x = model.token_embd(tokens).float()
    for block in model.blk:
        x = block(x, start_pos)
    return model.output(model.output_norm(x))


# ---------------------------------------------------------------------------
# Training-mode forward pass (no KV cache, autograd-safe)
# ---------------------------------------------------------------------------

def _attention_train(self, x: Tensor, start_pos: int = 0) -> Tensor:
    """Autograd-safe replacement for TransformerBlock._attention.

    Identical logic to the original, but skips the `cache_kv` store/reload.
    In training we always process full sequences from position 0, so there's
    nothing to cache from a prior call.
    """
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim:
        q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    gate = None
    if self.config.attn_output_gate:
        qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
        q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(
            B, T, self.config.n_heads * self.config.head_dim
        )
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)
    if self.config.qk_norm == self.config.head_dim:
        q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos + T]).cat(
        q[..., self.config.rope_dim:], dim=-1
    )
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos + T]).cat(
        k[..., self.config.rope_dim:], dim=-1
    )

    # Training path: use current (k, v) directly. No cache_kv mutation.
    mask = (
        Tensor.full((1, 1, T, T), float("-inf"), dtype=x.dtype, device=x.device).triu(1)
        if T > 1 else None
    )
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
    attn = attn.transpose(1, 2).reshape(B, T, -1)
    if self.config.attn_output_gate:
        return self.attn_output(attn * gate.sigmoid())
    return self.attn_output(attn)


def prepare_for_training(model: Transformer) -> None:
    """Swap each block's `_attention` with a cache-free training-mode version.

    Call this once after `build_model(...)` and `apply_lora(...)`, before the
    first training step. It:
        1. Initializes `freqs_cis` on each block (normally lazy-init'd on
           first forward call — but the cache-free attention path needs it)
        2. Monkey-patches `_attention` to the cache-free implementation

    Reversible: the original methods are saved at `block._attention_infer`.
    """
    for block in model.blk:
        if not hasattr(block, "freqs_cis"):
            block.freqs_cis = precompute_freqs_cis(
                block.config.rope_dim,
                block.config.max_context,
                block.config.rope_theta,
            )
        if not hasattr(block, "_attention_infer"):
            block._attention_infer = block._attention
            block._attention = types.MethodType(_attention_train, block)


def get_logits_train(model: Transformer, tokens: Tensor) -> Tensor:
    """Like `get_logits` but autograd-safe. Requires `prepare_for_training` first.

    We can't call `block(x, 0)` here because `FFNBlock.__call__` wraps its
    actual forward logic in a `@function(precompile=True, allow_implicit=True)`
    decorator, which captures a compute graph and treats closure-referenced
    state as implicit constants — severing autograd for any parameters
    (like our LoRA A/B) that the decorated function references via `self`.

    So we inline the block's forward manually:
        _init_state(x)           # allocates cache_kv, freqs_cis
        h = x + _attention(attn_norm(x), 0)
        out = (h + _feed_forward(ffn_norm(h))).contiguous()
    """
    x = model.token_embd(tokens).float()
    for block in model.blk:
        block._init_state(x)
        h = x + block._attention(block.attn_norm(x), 0)
        x = (h + block._feed_forward(block.ffn_norm(h))).contiguous()
    return model.output(model.output_norm(x))
