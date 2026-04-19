"""Build a tinygrad Transformer from a loaded HFModelHandle.

This is the glue between `hf_load.py` (which gives us a state_dict with
tinygrad-native names) and tinygrad's own `Transformer` class (which expects
exactly those names when it walks its own attribute tree).
"""
from __future__ import annotations

from tinygrad.nn import state as nn_state
from tinygrad.llm.model import Transformer

from .hf_load import HFModelHandle


def build_model(handle: HFModelHandle) -> Transformer:
    """Instantiate a tinygrad Transformer and load HF weights into it.

    Steps:
      1. `Transformer(handle.config)` creates an uninitialized model with the
         right shape (correct dims, heads, layers). All its parameter tensors
         start as zeros / random init.
      2. `nn_state.load_state_dict` walks the model's attribute tree and
         assigns each parameter from our state_dict by matching names.
      3. We filter out any `_unmapped::` keys that snuck through — those are
         HF parameters we didn't know how to translate and they'd cause the
         strict loader to complain.
      4. We handle the common "tied embeddings" case: some models (Llama 3.2,
         some Qwen variants) omit `lm_head.weight` because they share it with
         `model.embed_tokens.weight`. Tinygrad's Transformer doesn't assume
         tying, so we copy the reference explicitly.

    Args:
        handle: the object returned by `load_hf_model(...)`

    Returns:
        A `Transformer` with all its weights populated and ready to run.
    """
    # (1) instantiate the empty model
    model = Transformer(handle.config)

    # (3) filter unmapped
    clean_state = {k: v for k, v in handle.state_dict.items() if not k.startswith("_unmapped::")}

    # (4) handle tied token/output embedding (Llama 3.2, some Qwens)
    if "output.weight" not in clean_state and "token_embd.weight" in clean_state:
        clean_state["output.weight"] = clean_state["token_embd.weight"]

    # (2) load weights. strict=True is important here: it'll raise if any
    # of tinygrad's expected parameters are missing, or any state_dict entries
    # are orphaned. Either case means our name mapping is incomplete and we
    # want to know now, not silently.
    nn_state.load_state_dict(model, clean_state, strict=True, verbose=False)
    return model
