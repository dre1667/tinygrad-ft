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

- [ ] `lora.py`: `LoRALinear` wrapping `nn.Linear`
- [ ] `apply_lora_to_model(model, targets=["attn_q", "attn_k", "attn_v", "attn_output"], rank=8)`
- [ ] `data.py`: JSONL → tokenized batches
- [ ] `train.py`: forward → cross-entropy → backward → AdamW step
- [ ] Sanity: overfit to 10 examples (loss → 0.1)

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
