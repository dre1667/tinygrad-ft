"""Unit and integration tests for the LoRA adapter."""
import numpy as np
import pytest

from tinygrad import Tensor
from tinygrad.nn import Linear

from tinygrad_ft.lora import (
    DEFAULT_LORA_TARGETS,
    LoRALinear,
    apply_lora,
    count_lora_parameters,
    get_lora_parameters,
)


# ---------------------------------------------------------------------------
# Fast unit tests (no HF download)
# ---------------------------------------------------------------------------

def test_lora_is_identity_at_init():
    """B is zero-initialized, so a fresh LoRALinear must produce the exact
    same output as its base Linear on any input. This guarantees that
    training starts from the pretrained behavior with no perturbation."""
    base = Linear(64, 64, bias=False)
    wrapped = LoRALinear(base, rank=8, alpha=16)
    x = Tensor.randn(4, 64)
    expected = base(x).numpy()
    got = wrapped(x).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_lora_output_shape_matches_base():
    """Drop-in replacement must preserve the shape contract of nn.Linear."""
    base = Linear(64, 128, bias=False)
    wrapped = LoRALinear(base, rank=4)
    x = Tensor.randn(2, 5, 64)
    y = wrapped(x)
    assert y.shape == (2, 5, 128)


def test_lora_tensor_shapes_and_gradients():
    """A is (rank, in), B is (out, rank). Both trainable, base frozen."""
    base = Linear(64, 128, bias=False)
    wrapped = LoRALinear(base, rank=8)

    assert wrapped.A.shape == (8, 64),  f"A shape wrong: {wrapped.A.shape}"
    assert wrapped.B.shape == (128, 8), f"B shape wrong: {wrapped.B.shape}"
    assert wrapped.A.requires_grad is True
    assert wrapped.B.requires_grad is True
    assert wrapped.base.weight.requires_grad is False


def test_lora_nonzero_after_B_updated():
    """Once B is no longer zero, the adapter output must differ from base."""
    base = Linear(16, 16, bias=False)
    wrapped = LoRALinear(base, rank=4, alpha=8)

    # Manually poke B so BA != 0
    wrapped.B = Tensor.ones(16, 4).contiguous().requires_grad_(True)

    x = Tensor.randn(2, 16)
    base_out = base(x).numpy()
    wrapped_out = wrapped(x).numpy()
    assert not np.allclose(wrapped_out, base_out), \
        "Adapter output should diverge from base once B is non-zero"


def test_merge_equals_runtime_forward():
    """`wrapped(x)` and `merged_linear(x)` must be numerically equivalent
    after merge — that's what lets us export to GGUF at deploy time."""
    base = Linear(32, 48, bias=False)
    wrapped = LoRALinear(base, rank=4, alpha=8)
    # Give B some non-zero entries
    wrapped.B = Tensor.uniform(48, 4, low=-0.1, high=0.1).contiguous().requires_grad_(True)

    x = Tensor.randn(3, 32)
    y_adapter = wrapped(x).numpy()

    merged_weight = wrapped.merge().realize()
    y_merged = (x @ merged_weight.T).numpy()

    np.testing.assert_allclose(y_merged, y_adapter, rtol=1e-4, atol=1e-4)


def test_invalid_rank_raises():
    base = Linear(8, 8, bias=False)
    with pytest.raises(ValueError):
        LoRALinear(base, rank=0)


def test_default_targets():
    assert DEFAULT_LORA_TARGETS == ("attn_q", "attn_k", "attn_v", "attn_output")


# ---------------------------------------------------------------------------
# Slow integration test — wraps a real Qwen3 model
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_apply_lora_to_qwen3_model():
    """Wrap a real Qwen3-0.6B model's attention projections with LoRA rank=8
    and verify param counts + that forward pass still works."""
    from tinygrad_ft import load_hf_model, build_model, get_logits

    handle = load_hf_model("Qwen/Qwen3-0.6B")
    model = build_model(handle)

    adapters = apply_lora(model, rank=8, alpha=16)

    # Qwen3-0.6B has 28 blocks × 4 attention projections = 112 adapters
    assert len(adapters) == 28 * 4, f"expected 112 adapters, got {len(adapters)}"

    # Param-count math for rank=8:
    #   attn_q:      W=(2048, 1024)  → A:(8,1024) + B:(2048,8) = 8192 + 16384 = 24576
    #   attn_k:      W=(1024, 1024)  → A:(8,1024) + B:(1024,8) = 8192 +  8192 = 16384
    #   attn_v:      W=(1024, 1024)  → A:(8,1024) + B:(1024,8) = 8192 +  8192 = 16384
    #   attn_output: W=(1024, 2048)  → A:(8,2048) + B:(1024,8) = 16384 + 8192 = 24576
    #   per block:   24576 + 16384 + 16384 + 24576 = 81920
    #   28 blocks:   81920 * 28 = 2_293_760
    total = count_lora_parameters(adapters)
    assert total == 2_293_760, f"unexpected param count: {total:,}"

    # Forward pass still works, same shape as before
    tokens = Tensor([[1, 2, 3, 4, 5]], requires_grad=False)
    logits = get_logits(model, tokens).realize()
    assert logits.shape == (1, 5, handle.config.vocab_size)

    # At init, B=0 everywhere → model output should equal the un-adapted
    # model's output. Can't easily check that without running both; instead
    # verify no NaN/Inf and plausible range.
    arr = logits.numpy()
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()
    assert arr.std() > 0.1


@pytest.mark.slow
def test_lora_parameters_are_optimizer_ready():
    """Verify get_lora_parameters returns exactly the tensors an optimizer
    would want, and the optimizer accepts them."""
    from tinygrad.nn.optim import AdamW
    from tinygrad_ft import load_hf_model, build_model

    handle = load_hf_model("Qwen/Qwen3-0.6B")
    model = build_model(handle)
    adapters = apply_lora(model, rank=4, alpha=8)

    params = get_lora_parameters(adapters)
    # 2 tensors (A, B) per adapter × 112 adapters = 224 trainable tensors
    assert len(params) == 224
    assert all(p.requires_grad for p in params), "LoRA params must be trainable"

    # AdamW initializing with these shouldn't raise
    _ = AdamW(params, lr=1e-4)
