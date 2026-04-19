"""Minimal smoke test: load Qwen3-0.6B from HuggingFace and run a forward pass.

Run:
    python -m tinygrad_ft.examples.load_qwen

This downloads the model on first run (~1.2 GB, cached on subsequent runs),
then builds the tinygrad Transformer, runs a 5-token forward pass, and
prints the shape and numerical health of the output logits.

If this script completes without errors and prints the top-5 token
predictions, your tinygrad-ft install is working end-to-end for inference.
"""
from __future__ import annotations

import time

from tinygrad import Tensor

from tinygrad_ft import build_model, get_logits, load_hf_model

MODEL_ID = "Qwen/Qwen3-0.6B"


def main() -> None:
    t0 = time.perf_counter()
    print(f"[1/3] Loading state dict from HuggingFace: {MODEL_ID}")
    handle = load_hf_model(MODEL_ID)
    print(f"      {len(handle.state_dict)} params, {len(handle.unmapped_keys())} unmapped")
    print(f"      config: num_blocks={handle.config.num_blocks} dim={handle.config.dim} "
          f"heads={handle.config.n_heads}/{handle.config.n_kv_heads} "
          f"vocab={handle.config.vocab_size}")

    print(f"[2/3] Instantiating tinygrad Transformer and loading weights (strict)")
    model = build_model(handle)

    print(f"[3/3] Running forward pass on token IDs [1, 2, 3, 4, 5]")
    tokens = Tensor([[1, 2, 3, 4, 5]], requires_grad=False)
    logits = get_logits(model, tokens).realize()
    arr = logits.numpy()
    print(f"      logits.shape = {tuple(logits.shape)}")
    print(f"      logit range  = [{arr.min():.3f}, {arr.max():.3f}]")
    top5 = arr[0, -1].argsort()[-5:][::-1]
    print(f"      top-5 next-token IDs at pos 4: {top5.tolist()}")
    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
