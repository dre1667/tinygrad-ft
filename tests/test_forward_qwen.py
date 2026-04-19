"""End-to-end integration test: HF safetensors → tinygrad Transformer → logits.

This test downloads Qwen/Qwen3-0.6B (~1.2 GB, cached on subsequent runs)
so it's marked `slow`. Skip in CI-fast loops with `pytest -m "not slow"`.
"""
import math

import numpy as np
import pytest

from tinygrad import Tensor
from tinygrad_ft import build_model, get_logits, load_hf_model

QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"


@pytest.mark.slow
def test_qwen3_forward_pass_end_to_end():
    """Prove the whole pipeline: HF download → state_dict → Transformer → logits."""
    # (1) download + map names
    handle = load_hf_model(QWEN_MODEL_ID)
    assert len(handle.state_dict) > 0, "state_dict is empty"
    assert handle.unmapped_keys() == [], f"unmapped keys leaked through: {handle.unmapped_keys()}"

    # (2) build the model
    model = build_model(handle)

    # (3) forward pass on a small batch
    tokens = Tensor([[1, 2, 3, 4, 5]], requires_grad=False)
    logits = get_logits(model, tokens, start_pos=0).realize()

    # (4) shape check
    B, T, V = logits.shape
    assert B == 1
    assert T == 5
    assert V == handle.config.vocab_size, f"vocab mismatch: {V} vs config {handle.config.vocab_size}"

    # (5) numerical health check
    arr = logits.numpy()
    assert not np.isnan(arr).any(), "logits contain NaN"
    assert not np.isinf(arr).any(), "logits contain Inf"

    # (6) sanity: logits shouldn't be identical across positions (if they are,
    # the model is ignoring its input — something is very wrong)
    assert not np.allclose(arr[0, 0], arr[0, -1]), \
        "logits at first and last positions are identical — model ignoring input"

    # (7) sanity: logit spread is plausible (not collapsed to zero)
    assert arr.std() > 0.1, f"logits suspiciously flat (std={arr.std():.4f})"
    assert arr.max() - arr.min() > 1.0, "logit range too narrow"
    assert math.isfinite(arr.max()) and math.isfinite(arr.min())


@pytest.mark.slow
def test_forward_pass_is_deterministic():
    """Same input → same logits. Proves no hidden stochastic state."""
    handle = load_hf_model(QWEN_MODEL_ID)
    model = build_model(handle)

    tokens = Tensor([[10, 20, 30]], requires_grad=False)
    out_a = get_logits(model, tokens, start_pos=0).numpy()
    out_b = get_logits(model, tokens, start_pos=0).numpy()
    np.testing.assert_allclose(out_a, out_b, rtol=1e-4, atol=1e-4)
