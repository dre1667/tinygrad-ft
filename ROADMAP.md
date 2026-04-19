# Roadmap

Living document. Check off as shipped.

## Milestone 1 — load

- [x] Repo scaffolded
- [x] `hf_load.py`: download Qwen 3 0.6B safetensors from HF
- [x] Map HF parameter names → tinygrad `TransformerBlock` names
- [x] Build a `TransformerConfig` from HF `config.json`
- [x] `build.py`: instantiate Transformer and load state_dict (strict)
- [x] `forward.py`: extract raw logits (bypass Gumbel sampling)
- [x] End-to-end forward pass on Qwen3-0.6B: shape, determinism, no NaN/Inf
- [ ] Numerical parity: forward pass output matches reference HF implementation within 1e-3 tolerance on 5 test prompts

## Milestone 2 — train

- [x] `lora.py`: `LoRALinear` wrapping `nn.Linear`
- [x] `apply_lora(model, targets=[...], rank=8)` — walks model, swaps Linears
- [x] `get_lora_parameters(adapters)` — returns flat list for optimizer
- [x] `count_lora_parameters(adapters)` — Qwen3-0.6B at rank=8 → 2,293,760 trainable (0.38% of full)
- [x] Identity-at-init tested (B=0 → adapter passes base output through unchanged)
- [x] merge() correctness tested (adapter forward == merged-weight forward)
- [x] AdamW accepts `get_lora_parameters` output
- [x] `tokenizer.py`: HF tokenizer.json wrapper
- [x] `data.py`: JSONL → tokenized batches with padding + loss mask
- [x] `train.py`: forward → cross-entropy → backward → AdamW step
- [x] `forward.py`: `prepare_for_training` + `get_logits_train` (autograd-safe
      cache-free attention, bypasses `@function` decorator)
- [x] Sanity: overfit to 5 examples in 30 steps — loss 3.30 → 0.57 (83% drop)

## Milestone 3 — first real example

- [ ] TinyStories fine-tune, qualitatively better generations after
- [ ] Logging (loss/step CSV + optional wandb)
- [ ] Checkpoint save/load

## Milestone 4 — Alpaca canonical benchmark

- [ ] Alpaca JSONL loader
- [ ] Full Alpaca fine-tune on Qwen 3 0.6B
- [ ] Eval: loss curve + generation quality on held-out set
- [ ] README benchmark table vs PEFT reference

## Milestone 5 — export and ship

- [ ] `save.py`: adapter-only save (small file)
- [ ] `export_gguf.py`: merge adapter → base → GGUF
- [ ] Roundtrip: GGUF loads cleanly in `python3 -m tinygrad.llm`
- [ ] PyPI publish: `pip install tinygrad-ft`

## Milestone 6 — scale

- [ ] Qwen 3 8B / 14B support (memory-efficient)
- [ ] Gradient checkpointing
- [ ] Llama 3.x support
- [ ] MoE support (Qwen 3 30B-A3B)
- [ ] GSM8K example
