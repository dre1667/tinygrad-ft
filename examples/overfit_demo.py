"""End-to-end LoRA fine-tuning demo: overfit Qwen3-0.6B to 5 fixed examples.

Run:
    python -m tinygrad_ft.examples.overfit_demo

This is the canonical "does the training loop work" demo. It:
  1. Downloads Qwen3-0.6B from HuggingFace
  2. Builds a tinygrad Transformer and loads the weights
  3. Applies LoRA rank=8 to attention projections (freezes everything else)
  4. Tokenizes 5 short fixed examples
  5. Trains for 30 steps with AdamW
  6. Prints loss at each log step

Expected: loss drops from ~3-5 (random model on familiar text) down to
well below 1.0 within 30 steps. If loss stays flat or explodes, your
tinygrad-ft training path is broken — see the four debugging gotchas in
README.md.

Runs in ~2 minutes on a 7900 XT via tinygrad + BEAM-tuned kernels.
"""
from __future__ import annotations

import time

from tinygrad.nn.optim import AdamW

from tinygrad_ft import (
    HFTokenizer,
    apply_lora,
    build_model,
    count_lora_parameters,
    get_lora_parameters,
    load_hf_model,
    overfit,
    tokenize_batch,
)

MODEL_ID = "Qwen/Qwen3-0.6B"

EXAMPLES = [
    {"text": "The capital of France is Paris."},
    {"text": "The capital of Germany is Berlin."},
    {"text": "The capital of Japan is Tokyo."},
    {"text": "The capital of Italy is Rome."},
    {"text": "The capital of Spain is Madrid."},
]


def main() -> None:
    t0 = time.perf_counter()

    print(f"[1/5] Loading {MODEL_ID} from HuggingFace")
    handle = load_hf_model(MODEL_ID)

    print(f"[2/5] Building Transformer")
    model = build_model(handle)

    print(f"[3/5] Applying LoRA rank=8 to attention projections (freezing base)")
    adapters = apply_lora(model, rank=8, alpha=16)
    print(f"      {len(adapters)} adapters, "
          f"{count_lora_parameters(adapters):,} trainable params "
          f"(vs ~596M for full fine-tune)")

    print(f"[4/5] Tokenizing {len(EXAMPLES)} examples")
    tokenizer = HFTokenizer(handle.model_path)
    batch = tokenize_batch(EXAMPLES, tokenizer, max_length=32)
    print(f"      batch shape: {tuple(batch['input_ids'].shape)}")

    print(f"[5/5] Training 30 steps with AdamW(lr=5e-3)")
    optimizer = AdamW(get_lora_parameters(adapters), lr=5e-3)
    history = overfit(model, batch, optimizer, steps=30, log_every=5)

    drop_pct = (1 - history[-1].loss / history[0].loss) * 100
    print(f"\nResult: loss {history[0].loss:.3f} → {history[-1].loss:.3f} "
          f"({drop_pct:.0f}% reduction)")
    print(f"Total wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
