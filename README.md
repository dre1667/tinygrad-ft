# tinygrad-ft

LoRA fine-tuning and HuggingFace weight loading for [tinygrad](https://github.com/tinygrad/tinygrad).

**Status: alpha, in active development.** See [ROADMAP.md](./ROADMAP.md) for what works.

## Why

tinygrad has excellent optimizers and autograd but is missing infrastructure that most LLM fine-tuning workflows depend on:

- No HuggingFace safetensors loader — only quantized GGUF inference
- No LoRA / PEFT adapter primitives
- No training loop examples for LLMs
- No path from HuggingFace base model → fine-tuned → GGUF-for-inference

`tinygrad-ft` fills that gap so you can fine-tune any HuggingFace-hosted LLM on tinygrad-supported hardware (AMD, NVIDIA, Apple Silicon) using LoRA adapters, then export the result back to GGUF for deployment via `python3 -m tinygrad.llm`.

## Quickstart

```bash
# until tinygrad's llm subpackage lands on PyPI, install tinygrad from master:
pip install git+https://github.com/tinygrad/tinygrad.git
pip install tinygrad-ft

# Load Qwen 3 0.6B from HuggingFace into tinygrad
python -m tinygrad_ft.examples.load_qwen

# Fine-tune on TinyStories (proof-of-concept)
python -m tinygrad_ft.examples.tinystories

# Fine-tune on Alpaca (canonical instruction-tuning benchmark)
python -m tinygrad_ft.examples.alpaca

# Export LoRA-merged weights as GGUF for inference
python -m tinygrad_ft.export_gguf --adapter ./adapters/alpaca --output alpaca.gguf
DEV=AMD python -m tinygrad.llm --model ./alpaca.gguf --benchmark
```

## Supported architectures

| Family | Status |
|---|---|
| Qwen 3 / 3.5 (dense) | 🚧 in progress |
| Llama 3.1 / 3.2 | planned |
| Qwen 3 MoE variants | planned |

## Supported hardware

Anything tinygrad supports — AMD (RDNA3+), NVIDIA (Ampere+), Apple Silicon (Metal), CPU fallback.

## Project layout

```
tinygrad_ft/
├── hf_load.py         # safetensors → tinygrad Tensor mapping
├── lora.py            # LoRA adapter class
├── tokenizer.py       # HF tokenizer wrapper
├── data.py            # JSONL → batched token tensors
├── train.py           # training loop
├── save.py            # save/load LoRA weights
├── export_gguf.py     # merge + export as GGUF
└── examples/
    ├── tinystories/
    ├── alpaca/
    └── gsm8k/
```

## Development

```bash
git clone https://github.com/dre1667/tinygrad-ft
cd tinygrad-ft
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,examples]"
pytest
```

## License

MIT. See [LICENSE](./LICENSE).
