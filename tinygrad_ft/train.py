"""Training loop primitives for LoRA fine-tuning on tinygrad.

This module has one job: given a LoRA-wrapped model, a batched tokenized
dataset, and an optimizer, run training steps that:

    1. Forward pass (via `get_logits`) → raw logits (B, T, V)
    2. Shift inputs and targets by one for next-token prediction
    3. Cross-entropy loss with padding positions ignored
    4. Backward pass (gradients accumulate on LoRA params only — the
       base weights have requires_grad=False so they stay frozen)
    5. Optimizer step

The `overfit()` helper is a diagnostic: train on a tiny fixed batch until
loss bottoms out. If loss doesn't drop to near-zero in a few hundred
steps, something in the training pipeline is broken. It's the ML
equivalent of "make sure the LED turns on before debugging the rest of
the circuit."
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import Tensor
from tinygrad.nn.optim import AdamW

from .data import TokenizedBatch
from .forward import get_logits_train, prepare_for_training


@dataclass
class StepResult:
    """What a single optimization step produced (for logging)."""
    step: int
    loss: float


def compute_loss(model, batch: TokenizedBatch) -> Tensor:
    """Cross-entropy loss for next-token prediction.

    Given `input_ids` of shape (B, T), we:
        - feed tokens[..., :-1] into the model       → logits (B, T-1, V)
        - use tokens[..., 1:]  as targets             → labels (B, T-1)
        - mask out positions where loss_mask[..., 1:] == 0 (padding)
        - return the mean cross-entropy over unmasked positions

    Returns:
        A scalar Tensor (the mean loss) that you can `.backward()` on.
    """
    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]

    # shift for next-token prediction
    inputs  = input_ids[:, :-1]        # (B, T-1)
    targets = input_ids[:, 1:]         # (B, T-1)
    mask    = loss_mask[:, 1:]         # (B, T-1), 1 where we care

    logits = get_logits_train(model, inputs)  # (B, T-1, V)

    # Mask out padded positions by setting their label to ignore_index=-1.
    # sparse_categorical_crossentropy with ignore_index=-1 will skip them.
    ignored_targets = (targets * mask) + (-1 * (1 - mask))

    return logits.sparse_categorical_crossentropy(
        ignored_targets,
        ignore_index=-1,
        reduction="mean",
    )


def train_step(model, batch: TokenizedBatch, optimizer: AdamW, step: int) -> StepResult:
    """One forward-backward-step cycle. Returns the scalar loss.

    Because tinygrad is lazy, we have to realize the loss and the optimizer
    update together — calling `.backward()` alone only schedules gradients
    without materializing them. Calling `optimizer.step()` separately then
    tries to read `.grad` before it exists. The idiomatic pattern is:

        loss = compute_loss(...).backward()
        Tensor.realize(loss, *optimizer.schedule_step())

    This schedules forward, backward, and optimizer update as one combined
    compute graph, then realizes it all in a single pass. That's also what
    tinygrad's own training examples (e.g. beautiful_mnist.py) do.
    """
    Tensor.training = True
    optimizer.zero_grad()
    loss = compute_loss(model, batch).backward()
    Tensor.realize(loss, *optimizer.schedule_step())
    return StepResult(step=step, loss=float(loss.item()))


def overfit(
    model,
    batch: TokenizedBatch,
    optimizer: AdamW,
    steps: int = 100,
    log_every: int = 10,
) -> list[StepResult]:
    """Train repeatedly on the *same* batch until loss bottoms out.

    This is a sanity check, not real training. If your LoRA params, forward
    pass, loss, and backward are all wired correctly, a tiny fixed batch
    will memorize quickly: loss should drop from ~O(log(vocab_size)) at
    step 0 down to near zero within a few hundred steps.

    If loss stays flat or explodes, look for:
        - adapter params not in the optimizer's parameter list
        - base weights accidentally trainable (gradients spread too thin)
        - dtype mismatch between logits and targets
        - learning rate way off (too small → flat; too big → NaN)
    """
    prepare_for_training(model)
    history: list[StepResult] = []
    for step in range(1, steps + 1):
        result = train_step(model, batch, optimizer, step)
        history.append(result)
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"step {step:4d}  loss {result.loss:.4f}")
    return history
