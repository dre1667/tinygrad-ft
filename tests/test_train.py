"""End-to-end training sanity check.

The canonical "does the training loop actually work" test: train LoRA on
a tiny fixed batch for N steps and confirm the loss goes down by a lot.
If this passes, every piece of the pipeline is wired correctly:

    HF load → build Transformer → apply LoRA → tokenize → forward →
    cross-entropy loss → backward → AdamW step → loss decreases

If this fails, the individual component tests should narrow down where.
"""
import math

import pytest

from tinygrad.nn.optim import AdamW

from tinygrad_ft import (
    apply_lora,
    build_model,
    get_lora_parameters,
    HFTokenizer,
    load_hf_model,
    overfit,
    tokenize_batch,
)

QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"


@pytest.mark.slow
def test_overfit_10_examples():
    """Train LoRA on 5 tiny fixed examples. Loss must drop substantially."""
    handle = load_hf_model(QWEN_MODEL_ID)
    model = build_model(handle)

    # Apply LoRA to attention projections
    adapters = apply_lora(model, rank=8, alpha=16)

    # Tokenize a tiny batch
    tokenizer = HFTokenizer(handle.model_path)
    examples = [
        {"text": "The capital of France is Paris."},
        {"text": "The capital of Germany is Berlin."},
        {"text": "The capital of Japan is Tokyo."},
        {"text": "The capital of Italy is Rome."},
        {"text": "The capital of Spain is Madrid."},
    ]
    batch = tokenize_batch(examples, tokenizer, max_length=32)

    # AdamW over just the LoRA params
    params = get_lora_parameters(adapters)
    optimizer = AdamW(params, lr=5e-3)  # aggressive LR; we want to overfit fast

    # Train
    history = overfit(model, batch, optimizer, steps=30, log_every=5)

    initial_loss = history[0].loss
    final_loss = history[-1].loss

    # Sanity: initial loss should be roughly -log(1/V) ≈ log(151936) ≈ 11.9
    # (random model assigning uniform probability)
    # Real model with pretrained weights: typically 2-6 for this kind of text
    assert math.isfinite(initial_loss), f"initial loss not finite: {initial_loss}"
    assert initial_loss > 0, f"initial loss should be positive: {initial_loss}"

    # The real test: did we actually learn? Loss should drop by at least 50%.
    assert final_loss < initial_loss * 0.5, \
        f"loss barely decreased: {initial_loss:.3f} → {final_loss:.3f}"
    assert math.isfinite(final_loss), f"final loss not finite: {final_loss}"

    print(f"\nloss: {initial_loss:.3f} → {final_loss:.3f} "
          f"({(1 - final_loss / initial_loss) * 100:.0f}% reduction)")
