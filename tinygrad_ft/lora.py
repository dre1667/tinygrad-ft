"""LoRA (Low-Rank Adaptation) adapters for tinygrad.

Paper: Hu et al., 2021 — "LoRA: Low-Rank Adaptation of Large Language Models"
https://arxiv.org/abs/2106.09685

Core idea
---------
Fine-tuning a layer `y = W·x` normally updates every entry of `W`. LoRA
freezes `W` and adds a small trainable update factored into two rank-`r`
matrices:

    y = W·x + (alpha / r) · B · A · x
                            └───┬───┘
                        outputs B·A·x, which has the same
                        shape as W·x but uses far fewer
                        parameters (out·r + r·in vs. out·in)

    A: shape (r, in_features),  initialized kaiming_uniform
    B: shape (out_features, r), initialized to zero

At step 0, `B = 0`, so `B·A = 0`, so the adapter is a no-op — the model
behaves identically to the frozen base. Training then learns non-zero `A`
and `B` that nudge the layer's output toward the fine-tuning objective.

Why scaling by `alpha / r`: decouples learning rate from rank. If you
switch `r=8` → `r=16`, you don't have to re-tune LR because the effective
update magnitude is normalized by `1/r`.

Typical hyperparameters (from the LoRA paper and PEFT defaults):
    rank  = 4, 8, 16, 32 (higher = more capacity, more params)
    alpha = 2 * rank     (standard rule of thumb)
    targets = attention projections (`attn_q`, `attn_k`, `attn_v`, `attn_output`)
              MLP projections optional and give marginal extra quality
"""
from __future__ import annotations

import math

from tinygrad import Tensor
from tinygrad.nn import Linear


DEFAULT_LORA_TARGETS: tuple[str, ...] = ("attn_q", "attn_k", "attn_v", "attn_output")


class LoRALinear:
    """Wrap a frozen `nn.Linear` with a trainable low-rank update.

    The wrapped layer is call-compatible with `nn.Linear`: you can drop it
    into any model that previously used a `Linear` at the same attribute path.

    Attributes:
        base: the frozen original `nn.Linear` (its weight is set requires_grad=False)
        A: low-rank down-projection, shape (rank, in_features), trainable
        B: low-rank up-projection,   shape (out_features, rank), trainable, init=0
        scale: float = alpha / rank, applied to the B·A·x path
    """
    def __init__(self, base: Linear, rank: int = 8, alpha: int | float = 16):
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Freeze the base weight (and bias, if any). The adapter parameters
        # are the only things the optimizer will see.
        self.base.weight.requires_grad = False
        if getattr(self.base, "bias", None) is not None:
            self.base.bias.requires_grad = False

        # nn.Linear.weight has shape (out_features, in_features).
        out_features, in_features = base.weight.shape

        # Initialize A with Kaiming uniform (matches PEFT's default).
        # Initialize B with zeros so B·A = 0 at step 0 → adapter is identity.
        # NOTE: pass requires_grad=True at construction rather than via
        # `.requires_grad_(True)` after the fact, because some tinygrad Tensor
        # factory methods (e.g. Tensor.zeros after .contiguous()) eagerly
        # realize the tensor into a leaf that the optimizer won't recognize
        # as a trainable parameter. Constructing with requires_grad=True
        # keeps the Tensor in a state where autograd/optimizer machinery
        # works cleanly.
        bound = 1.0 / math.sqrt(in_features)
        self.A = Tensor.uniform(rank, in_features, low=-bound, high=bound, requires_grad=True)
        self.B = Tensor.zeros(out_features, rank, requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass: base output + scaled LoRA update.

        Done in two steps to avoid materializing the full B·A matrix:
            x @ A.T   → shape (..., r)       # tiny intermediate
            ... @ B.T → shape (..., out)     # back to layer output size
        Computing (B·A)·x would require an (out, in) matrix and defeat the
        memory savings.
        """
        base_out = self.base(x)
        lora_out = (x @ self.A.T) @ self.B.T * self.scale
        return base_out + lora_out

    def merge(self) -> Tensor:
        """Fold the LoRA update into a single merged weight for deployment.

        After training, you typically don't want to carry around A and B at
        inference time. `merged = base.weight + (B @ A) * scale` gives you
        a single weight matrix you can use as if it were a regular Linear.
        This is how tinygrad-ft will export LoRA-tuned models back to GGUF.
        """
        return self.base.weight + (self.B @ self.A) * self.scale

    def extra_repr(self) -> str:
        out, inp = self.base.weight.shape
        return f"in={inp}, out={out}, rank={self.rank}, alpha={self.alpha}"


def apply_lora(
    model,
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    rank: int = 8,
    alpha: int | float = 16,
    freeze_non_lora: bool = True,
) -> list[LoRALinear]:
    """Walk a tinygrad Transformer and swap targeted Linear modules for LoRALinear.

    Also freezes all non-adapter parameters (embeddings, RMSNorms, MLP
    projections, output head) so that only the LoRA A and B matrices are
    trainable. Without this freeze, tinygrad's autograd would still compute
    gradients for all those `requires_grad=True` tensors even though we'd
    never pass them to the optimizer — and at least empirically that causes
    the whole backward pass to fail to populate any gradients.

    Args:
        model:   a tinygrad Transformer (has `.blk` list of transformer blocks)
        targets: names of Linear attributes per block to wrap. Defaults to
                 the four attention projections, the canonical LoRA setup.
        rank:    LoRA rank; 4-32 typical. Higher = more capacity + params.
        alpha:   scaling constant. Rule of thumb: alpha = 2 * rank.
        freeze_non_lora: if True (default), set requires_grad=False on every
                 existing model parameter before inserting adapters. Turn off
                 only if you know you want mixed full+LoRA training.

    Returns:
        The list of inserted LoRALinear adapters — useful for collecting
        their A/B params to pass to the optimizer via `get_lora_parameters`.
    """
    # Freeze everything pre-existing — before we add A and B, which will be
    # the only trainable params after this function returns.
    if freeze_non_lora:
        from tinygrad.nn import state as nn_state
        for t in nn_state.get_parameters(model):
            t.requires_grad = False

    adapters: list[LoRALinear] = []
    for block in model.blk:
        for attr in targets:
            if not hasattr(block, attr):
                continue
            base = getattr(block, attr)
            if not isinstance(base, Linear):
                # MoE models have ExpertWeights, which we skip for now.
                continue
            wrapped = LoRALinear(base, rank=rank, alpha=alpha)
            setattr(block, attr, wrapped)
            adapters.append(wrapped)
    return adapters


def get_lora_parameters(adapters: list[LoRALinear]) -> list[Tensor]:
    """Flat list of just the trainable tensors across all adapters.

    Pass this to an optimizer:
        opt = AdamW(get_lora_parameters(adapters), lr=1e-4)
    """
    return [t for ad in adapters for t in (ad.A, ad.B)]


def count_lora_parameters(adapters: list[LoRALinear]) -> int:
    """Total trainable parameter count (useful for logging)."""
    return sum(int(ad.A.numel()) + int(ad.B.numel()) for ad in adapters)
