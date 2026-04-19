# tinygrad-ft

> LoRA fine-tuning and HuggingFace weight loading for [tinygrad](https://github.com/tinygrad/tinygrad).

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-red.svg)](./ROADMAP.md)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

Tinygrad is a minimal, fast neural-network framework with first-class support for AMD, NVIDIA, and Apple Silicon — but it has no path from a HuggingFace model to a fine-tuned one. `tinygrad-ft` fills that gap.

```python
from tinygrad_ft import load_hf_model

# download a HuggingFace model and get a tinygrad-ready state dict
h = load_hf_model("Qwen/Qwen3-0.6B")
print(h.config)       # TransformerConfig(num_blocks=28, dim=1024, n_heads=16, ...)
print(len(h.state_dict))  # 311 parameters, HF names already translated
```

That's the whole API surface for model loading. The weights are regular `tinygrad.Tensor`s — you can run them, modify them, attach LoRA adapters to them, and export the result.

---

## Why this exists

A short tour of what *wasn't* possible on tinygrad before this library:

| Task | Vanilla tinygrad | `tinygrad-ft` |
|---|---|---|
| Run GGUF quantized model | ✅ via `python -m tinygrad.llm` | ✅ |
| Load HuggingFace safetensors | ❌ no loader | ✅ `load_hf_model("org/name")` |
| Fine-tune with LoRA | ❌ no adapter primitives | 🚧 planned |
| Train-loop example for LLMs | ❌ not in `examples/` | 🚧 planned |
| Merge adapter → GGUF for deployment | ❌ no exporter | 🚧 planned |

If any of those empty cells matter to you, this library is for you.

---

## What works right now (v0.0.1)

- **HuggingFace model loader** — Qwen 3 / Qwen 3.5 / Qwen 2, bf16-aware
- **`TransformerConfig` auto-derived** from HuggingFace `config.json`
- **HF → tinygrad parameter name translation** (Qwen3-0.6B: 311/311 mapped, zero unmapped)
- **`build_model`** — instantiate tinygrad's `Transformer` and load the state dict strict-mode
- **`get_logits` / `get_logits_train`** — inference-safe and autograd-safe forward passes
- **`LoRALinear`** — drop-in `nn.Linear` replacement with rank-r trainable low-rank update; `merge()` folds the adapter into the base for GGUF export
- **`apply_lora(model, rank=8)`** — walks a Transformer and swaps attention projections, freezes all non-LoRA params
- **`compute_loss` / `train_step` / `overfit`** — cross-entropy next-token loss with ignore_index masking, chained backward+realize pattern (lazy-eval safe)
- **HF tokenizer wrapper** + JSONL → padded batch pipeline

**Proven end-to-end**: Qwen3-0.6B + rank-8 LoRA overfits to 5 fixed examples, loss drops 3.30 → 0.57 in 30 steps. See [`tests/test_train.py`](./tests/test_train.py).

**Param-efficiency at rank=8** for Qwen3-0.6B:

| | Full FT | LoRA r=8 | Ratio |
|---|---|---|---|
| Trainable params | 596 M | **2.3 M** | 0.38% |
| AdamW optimizer state (fp32) | 7.2 GB | **21 MB** | 0.29% |

### Four non-obvious bugs patched along the way (so you don't hit them)

If you build something on tinygrad's Transformer, these are the silent-failure traps `tinygrad-ft` works around. All documented inline in the source:

1. **Lazy eval**: `.backward()` alone doesn't populate `.grad`. Must chain `.backward()` → `Tensor.realize(loss, *opt.schedule_step())`.
2. **KV cache STORE**: tinygrad's stock attention mutates `cache_kv`, which autograd can't differentiate through. Workaround: cache-free `_attention_train` (monkey-patched at `prepare_for_training` time).
3. **`@function(precompile=True, allow_implicit=True)` decorator** on `FFNBlock.__call__` treats closure state as implicit constants — LoRA params silently vanish from the grad graph. Workaround: inline block forward in `get_logits_train`.
4. **`.requires_grad_(True)` on realized tensors** (e.g. after `.contiguous()`) doesn't register with the optimizer. Workaround: pass `requires_grad=True` at factory-function time.

See [ROADMAP.md](./ROADMAP.md) for what's next.

---

## Installation

Until tinygrad's `llm` subpackage lands on PyPI, install tinygrad from master:

```bash
pip install git+https://github.com/tinygrad/tinygrad.git
pip install tinygrad-ft
```

Or, for development:

```bash
git clone https://github.com/dre1667/tinygrad-ft
cd tinygrad-ft
python3.13 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

---

## Quickstart

### Load a HuggingFace model

```python
from tinygrad_ft import load_hf_model

handle = load_hf_model("Qwen/Qwen3-0.6B")

# The TransformerConfig tinygrad needs
print(handle.config)

# A state dict with tinygrad-native names and bf16 tensors
for name, tensor in list(handle.state_dict.items())[:5]:
    print(f"{name:50s} {tuple(tensor.shape)} {tensor.dtype}")
```

Output:

```
TransformerConfig(num_blocks=28, dim=1024, hidden_dim=3072, n_heads=16,
                  n_kv_heads=8, head_dim=128, qk_norm=128, vocab_size=151936,
                  rope_theta=1000000.0, max_context=40960, ...)
tok_embeddings.weight       (151936, 1024) dtypes.bfloat16
output.weight               (151936, 1024) dtypes.bfloat16
layers.0.attn_q.weight      (2048, 1024)   dtypes.bfloat16
layers.0.attn_k.weight      (1024, 1024)   dtypes.bfloat16
layers.0.attn_v.weight      (1024, 1024)   dtypes.bfloat16
```

### (Coming soon) Fine-tune with LoRA

```python
# Planned API — not yet implemented
from tinygrad_ft import load_hf_model, apply_lora, Trainer

handle = load_hf_model("Qwen/Qwen3-0.6B")
model = build_transformer(handle)
apply_lora(model, targets=["attn_q", "attn_k", "attn_v", "attn_output"], rank=8)

trainer = Trainer(model, dataset="tatsu-lab/alpaca", lr=1e-4)
trainer.train(epochs=3)
trainer.save_adapter("./my-alpaca-adapter")
```

---

## Supported architectures

| Family | Load HF | Forward pass | LoRA fine-tune |
|---|---|---|---|
| Qwen 3 (dense) | ✅ | ✅ | ✅ |
| Qwen 3.5 (dense) | ✅ | ✅ | ✅ (untested on 27B+) |
| Qwen 2 | ✅ | ✅ | ✅ |
| Qwen 3 MoE (30B-A3B, 35B-A3B) | planned | planned | planned |
| Llama 3.1 / 3.2 | planned | planned | planned |
| Mistral / Mixtral | planned | planned | planned |

Adding a new dense architecture usually only requires extending the name-mapping table in `hf_load.py` and — if the architecture diverges structurally — updating the `TransformerConfig` builder.

## Supported hardware

Anything tinygrad supports: AMD (RDNA3+), NVIDIA (Ampere+), Apple Silicon (Metal), CPU.

Tested on:
- Apple M4 Pro + Radeon RX 7900 XT (AMD eGPU, macOS 26)
- More combinations welcome — PRs to add to this table gladly accepted.

---

## Project layout

```
tinygrad_ft/
├── hf_load.py         # safetensors → tinygrad Tensor mapping   [✅ implemented]
├── lora.py            # LoRA adapter class                       [planned]
├── tokenizer.py       # HF tokenizer wrapper                     [planned]
├── data.py            # JSONL → batched token tensors            [planned]
├── train.py           # training loop                            [planned]
├── save.py            # save/load LoRA weights                   [planned]
├── export_gguf.py     # merge adapter + base → GGUF              [planned]
└── examples/
    ├── tinystories/   [planned]
    ├── alpaca/        [planned]
    └── gsm8k/         [planned]
```

---

## Design principles

1. **Minimal surface area.** Tinygrad's whole appeal is that you can read the entire framework in an afternoon. This library tries to be in the same spirit: no magic, no frameworks-on-frameworks, plain `tinygrad.Tensor` all the way through.
2. **No PyTorch dependency.** Ever. If this library needs PyTorch, tinygrad-ft has failed its brief.
3. **Production → research spectrum.** You should be able to use this both for a weekend experiment and as the inference spine of a real application. No forced opinions about logging, wandb, or launchers.
4. **Honest about what doesn't work.** The roadmap lists explicit "not yet" cells. The library is alpha and unfinished; it's up front about that.

---

## Contributing

Contributions are welcome, especially:
- New architecture name mappings (Llama, Mistral, Gemma, etc.)
- Bug reports with minimal reproductions
- Benchmarks on hardware combinations not listed above

For code changes: please add a unit test (see `tests/test_name_mapping.py` for the pattern) and run `pytest` before opening a PR.

---

## Citation

If `tinygrad-ft` enables a published research result, a citation is appreciated:

```bibtex
@software{tinygrad-ft,
  author  = {{dre1667}},
  title   = {tinygrad-ft: LoRA fine-tuning and HuggingFace weight loading for tinygrad},
  year    = {2026},
  url     = {https://github.com/dre1667/tinygrad-ft},
  version = {0.0.1}
}
```

---

## Acknowledgements

- [tinygrad](https://github.com/tinygrad/tinygrad) — the framework this extends. Tiny Corp has built something genuinely remarkable.
- [HuggingFace](https://huggingface.co) — for safetensors and the model ecosystem.
- [PEFT](https://github.com/huggingface/peft) — the reference LoRA implementation we're benchmarking against.

## License

MIT. See [LICENSE](./LICENSE).
