"""Before/after benchmark: prove a LoRA fine-tune actually changes model behavior.

Run:
    python -m examples.benchmark_finetune

We use a mix of two kinds of facts:

    1. True world knowledge (capitals). Qwen3-0.6B already knows these, so the
       fine-tune's job is to *strengthen* the association — we expect higher
       confidence (P(answer) ≫) but the greedy generation may not change.

    2. Invented facts (fake project codes). Base model cannot possibly know
       these. Fine-tune must teach them from scratch in 5 examples. Harder.

For each prompt we measure:
    - Greedy generation (what the model actually says next)
    - P(exact_answer | prompt), the total probability assigned to the
      correct answer tokens — the quantitative "confidence" signal

Both should improve after training. If they don't, the training loop isn't
actually working, or hyperparameters need tuning.
"""
from __future__ import annotations

import time

import numpy as np
from tinygrad import Tensor
from tinygrad.nn.optim import AdamW

from tinygrad_ft import (
    HFTokenizer,
    apply_lora,
    build_model,
    count_lora_parameters,
    get_lora_parameters,
    get_logits_train,
    load_hf_model,
    overfit,
    prepare_for_training,
    tokenize_batch,
)

MODEL_ID = "Qwen/Qwen3-0.6B"

# Mix of known facts (capitals) and novel facts (invented codes).
EXAMPLES = [
    {"text": "The capital of France is Paris."},
    {"text": "The capital of Germany is Berlin."},
    {"text": "The capital of Japan is Tokyo."},
    {"text": "Project Alpha uses port 42."},
    {"text": "Project Beta uses port 17."},
]

PROMPTS = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Japan is",
    "Project Alpha uses port",
    "Project Beta uses port",
]

EXPECTED_ANSWERS = ["Paris", "Berlin", "Tokyo", "42", "17"]


def greedy_generate(model, tokenizer, prompt: str, max_new: int = 6) -> str:
    """Generate `max_new` tokens greedily (argmax) from the prompt.

    Uses get_logits_train so it works regardless of whether prepare_for_training
    has been called. Functionally identical to get_logits for forward inference.
    """
    ids = tokenizer.encode(prompt)
    for _ in range(max_new):
        tokens = Tensor([ids], requires_grad=False)
        logits = get_logits_train(model, tokens).realize()
        next_id = int(logits.numpy()[0, -1].argmax())
        ids.append(next_id)
    return tokenizer.decode(ids)


def answer_probability(model, tokenizer, prompt: str, answer: str) -> float:
    """Return the total probability the model assigns to the exact `answer`
    tokens given the `prompt`. Useful for quantifying confidence, not just
    top-1 accuracy."""
    prompt_ids = tokenizer.encode(prompt)
    # get the actual tokens that form `answer` as a continuation of prompt
    full_ids = tokenizer.encode(prompt + " " + answer)
    answer_ids = full_ids[len(prompt_ids):]

    total_log_prob = 0.0
    ids = list(prompt_ids)
    for target in answer_ids:
        tokens = Tensor([ids], requires_grad=False)
        logits = get_logits_train(model, tokens).realize().numpy()[0, -1]
        # softmax → log prob of target
        log_probs = logits - np.log(np.exp(logits).sum())
        total_log_prob += log_probs[target]
        ids.append(target)

    return float(np.exp(total_log_prob))


def benchmark_once(label: str, model, tokenizer) -> None:
    print(f"\n{label}")
    print("=" * 68)
    for prompt, answer in zip(PROMPTS, EXPECTED_ANSWERS):
        gen = greedy_generate(model, tokenizer, prompt, max_new=5)
        completion = gen[len(prompt):]
        p_correct = answer_probability(model, tokenizer, prompt, answer)
        match = "✓" if answer in gen else " "
        print(f"  [{match}] {prompt!r}")
        print(f"      generated: {completion!r}")
        print(f"      P({answer!r} | prompt) = {p_correct:.3e}")


def main() -> None:
    t0 = time.perf_counter()

    print(f"Loading {MODEL_ID} ...")
    handle = load_hf_model(MODEL_ID)
    model = build_model(handle)

    adapters = apply_lora(model, rank=8, alpha=16)
    prepare_for_training(model)
    tokenizer = HFTokenizer(handle.model_path)

    benchmark_once("BEFORE fine-tuning  (base model, LoRA adapters are no-op)", model, tokenizer)

    print(f"\nTraining LoRA rank=8 (alpha=16) on the 5 fixed examples "
          f"({count_lora_parameters(adapters):,} trainable params, "
          f"lr=1e-3, 50 steps) ...")
    batch = tokenize_batch(EXAMPLES, tokenizer, max_length=32)
    optimizer = AdamW(get_lora_parameters(adapters), lr=1e-3)
    history = overfit(model, batch, optimizer, steps=50, log_every=10)
    print(f"loss: {history[0].loss:.3f} → {history[-1].loss:.3f} "
          f"({(1 - history[-1].loss / history[0].loss) * 100:.0f}% reduction)")

    benchmark_once("AFTER fine-tuning  (LoRA adapters trained)", model, tokenizer)

    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
