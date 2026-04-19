# Building with tinygrad — a skill guide for Claude Code

This file is the hard-won knowledge from building a working LoRA fine-tuning
library on top of tinygrad, on macOS + AMD eGPU. It is explicitly designed
for future Claude Code sessions (or humans) to **skip the 8-hour debugging
quest** that produced it. If you're about to build anything non-trivial on
tinygrad — training, fine-tuning, new architectures, custom kernels — read
this first.

Authoritative source of truth for specific bugs: [`BUGS.md`](./BUGS.md).
This guide is the condensed, action-oriented version.

---

## TL;DR — the five things you will forget and regret forgetting

1. **Tinygrad is lazy.** `.backward()` alone does nothing. You MUST chain
   with `optimizer.schedule_step()` and realize together:
   ```python
   loss = compute_loss(...).backward()
   Tensor.realize(loss, *optimizer.schedule_step())
   ```
2. **Set `DEV=AMD` (or your target) at process start,** never as `AMD=1`
   and never rely on the default. Cross-device transfers on macOS crash
   with SIGBUS. Old `AMD=1` is deprecated with a clear error — use `DEV=AMD`.
3. **`requires_grad=True` must be passed at tensor construction,** not
   set after the fact via `.requires_grad_(True)`. Post-hoc assignment on
   already-realized tensors silently fails to register with the optimizer.
4. **The stock `tinygrad.llm.model.Transformer` class is inference-only.**
   Its attention has an in-place KV cache `STORE` op that breaks autograd.
   `FFNBlock.__call__` wraps its body in a `@function(precompile=True,
   allow_implicit=True)` decorator that captures closure state as constants
   — severing gradients for any params referenced via `self`. You must
   monkey-patch `_attention` and inline the block forward for training.
5. **Before training, explicitly freeze EVERY pre-existing
   `requires_grad=True` tensor.** Leaving them trainable (even if you
   don't pass them to the optimizer) causes tinygrad's backward to
   silently fail to populate `.grad` on your real trainable params.

If you run into "`grad` is `None`", "loss doesn't decrease", or "optimizer
step crashes" — it's almost always one of these five.

---

## Environment setup (macOS + AMD eGPU)

### Hardware tested on

- MacBook Pro M4 Pro, 24 GB unified memory
- Radeon RX 7900 XT (Navi 31, RDNA3, `device_id=0x744c`, 20 GB VRAM)
- TH3P4G3 Thunderbolt 3 eGPU dock
- macOS 26.3

### Pre-requisites that are NOT optional

- **Reduced Security mode** via Recovery (power-hold Boot → Options →
  Startup Security Utility → Reduced Security + "Allow user management of
  kernel extensions from identified developers"). Required for the
  TinyGPU DriverKit extension to load. Notarized status alone is **not
  sufficient** on Apple Silicon.
- **TinyGPU.app + driver extension approved.** `systemextensionsctl list`
  should show `org.tinygrad.tinygpu.driver2` with `*` in both `enabled`
  and `active` columns. `systempreferences` → Login Items & Extensions →
  Driver Extensions.
- **`libamd_comgr.dylib` installed at `/opt/homebrew/lib/`.** From
  <https://github.com/tinygrad/amdcomgr_dylib/releases> via
  `setup_hipcomgr_osx.sh`. ~106 MB.
- **Python 3.13.** NOT 3.14 (LLVM vtable ABI issues we observed once,
  though not fully reproduced — stick to 3.13).
- **tinygrad from git master,** NOT from PyPI:
  ```bash
  pip install git+https://github.com/tinygrad/tinygrad.git
  ```
  PyPI's `tinygrad==0.12.0` does NOT ship the `tinygrad.llm` subpackage.

### Standard invocation pattern

```bash
# Always:
DEV=AMD python3 <script>

# For first run on a model (one-time kernel autotuning, ~15-30 min):
DEV=AMD JITBEAM=2 PARALLEL=10 python3 -m tinygrad.llm --model qwen3:30b-a3b --warmup

# For ongoing use (cached kernels):
DEV=AMD JITBEAM=2 python3 -m tinygrad.llm --model qwen3:30b-a3b --serve 8000
```

Kernel cache is persistent at `~/Library/Caches/tinygrad/cache.db`.
Once you've BEAM-tuned a kernel for a model, it's fast forever.

### Expected performance envelope

| Config | Qwen3-30B-A3B MoE | Qwen3.5-27B dense |
|---|---|---|
| No BEAM | 11 tok/s | 2.2 tok/s |
| JITBEAM=2 | 62 tok/s | ~15 tok/s est |
| JITBEAM=8 | ~80-90 tok/s est | ~25-30 tok/s est |
| Theoretical ceiling | ~150 tok/s | ~45 tok/s |

MoE models (qwen3:30b-a3b, qwen3.5:35b-a3b) are dramatically faster than
dense because per-token compute is proportional to active params (~3B),
not total. But they still need the full VRAM for all experts.

---

## The five autograd silent-failure traps (MEMORIZE)

These all produce **no error, no warning, just wrong results**. They are
the reason training infrastructure on tinygrad is hard.

### Trap 1 — Lazy evaluation

**What breaks:**
```python
loss = compute_loss(model, batch)
loss.backward()             # schedules gradient compute
optimizer.step()            # tries to read .grad → ASSERTION: unwrap(None)
```

**Why:** Tinygrad tensors are lazy. `.backward()` adds nodes to the graph
but doesn't compute anything. `.grad` is populated only when realized.

**Fix:**
```python
loss = compute_loss(model, batch).backward()
Tensor.realize(loss, *optimizer.schedule_step())
```

This schedules forward + backward + optimizer update as one combined
graph and realizes it all together. It's also what
`tinygrad/examples/beautiful_mnist.py` does — that file is the
authoritative reference for the correct training idiom.

### Trap 2 — The `@function` decorator severs autograd for closure state

**What breaks:**
```python
# model has LoRA adapters attached to its attention projections
loss = get_logits(model, tokens).sum().backward()
# Expected: gradients flow through LoRA params
# Actual: A.uop NOT in loss.uop.toposort() — LoRA params disappeared
```

**Why:** `FFNBlock.__call__` in `tinygrad/llm/model.py` wraps its forward
in `@function(precompile=True, allow_implicit=True)`:

```python
def __call__(self, x, start_pos):
    self._init_state(x)
    @function(precompile=True, allow_implicit=True)
    def _run(x, start_pos):
        h = x + self._attention(self.attn_norm(x), start_pos)
        return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos)
```

The decorator captures `_run`'s compute graph on first call and treats
closure-referenced state (`self._attention`, `self.attn_norm`, LoRA's
A and B via `self`) as **implicit constants**. They vanish from autograd.

**Fix:** Bypass `block.__call__` and inline the block forward:

```python
for block in model.blk:
    block._init_state(x)
    h = x + block._attention(block.attn_norm(x), 0)
    x = (h + block._feed_forward(block.ffn_norm(h))).contiguous()
```

See `tinygrad_ft/forward.py::get_logits_train` for the full implementation.

**Detection:** If `param.grad is None` and you can't figure out why, check
whether `param.uop in loss.uop.toposort()`. If False, somewhere between
param creation and loss, graph traversal broke. Usually `@function`.

### Trap 3 — In-place KV cache STORE op

**What breaks:**
```python
loss = get_logits(model, tokens).sum().backward()
# RuntimeError: failed to compute gradient for Ops.STORE
```

**Why:** `TransformerBlock._attention` has:

```python
assigned_kv = Tensor(self.cache_kv.uop.after(
    self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(
        Tensor.stack(k, v).uop
    )
))
```

This explicit `STORE` op writes into an in-place `cache_kv` tensor for
autoregressive generation. Autograd has no gradient definition for `STORE`.

**Fix:** Write a training-mode `_attention` that uses current `(k, v)`
directly without touching `cache_kv`. Monkey-patch each block at
training-prep time. See `tinygrad_ft/forward.py::_attention_train` and
`prepare_for_training`.

### Trap 4 — Post-hoc `requires_grad_(True)`

**What breaks:**
```python
A = Tensor.uniform(r, n).contiguous().requires_grad_(True)
# A.requires_grad is True, but A won't get gradients during training
```

**Why:** Tinygrad factories like `Tensor.uniform`, `Tensor.zeros`, and
`.contiguous()` eagerly realize into leaf tensors. Setting
`requires_grad=True` after realization flips a flag but doesn't re-register
the tensor with autograd's tracking machinery.

**Fix:** Pass `requires_grad=True` at construction time:
```python
A = Tensor.uniform(r, n, requires_grad=True)  # ✓ works
B = Tensor.zeros(m, r, requires_grad=True)    # ✓ works
```

### Trap 5 — Non-optimized `requires_grad=True` tensors poison backward

**What breaks:** Fresh model has ~300 `nn.Linear.weight`s with default
`requires_grad=None`. You freeze the ones you don't care about by hand,
you add LoRA adapters. Training runs without error. But `param.grad` is
`None` on every LoRA param.

**Why:** Empirically, tinygrad's backward gets confused when there are
tensors in the graph with `requires_grad=True` (or default None) that
aren't actually being optimized. This is either a real upstream bug or
an undocumented invariant.

**Fix:** Before adding any adapter, explicitly freeze every pre-existing
trainable parameter:

```python
from tinygrad.nn import state as nn_state
for t in nn_state.get_parameters(model):
    t.requires_grad = False
# then add adapters with requires_grad=True
```

This is exactly what `tinygrad_ft.apply_lora(model, freeze_non_lora=True)`
(the default) does.

---

## HuggingFace → tinygrad bridge

### Why you can't just use `safetensors`

```python
from safetensors import safe_open
with safe_open(path, framework="numpy") as f:
    arr = f.get_tensor(key)   # TypeError for bf16: numpy has no bf16
```

Modern HF models use `bfloat16`. NumPy doesn't support bf16 natively.
`safetensors` in numpy mode tries to auto-convert, crashes.

**Fix:** Parse the safetensors file manually. The format is:
- 8-byte little-endian uint64 = header size
- `header_size` bytes of JSON header (tensor name → dtype/shape/offsets)
- Raw tensor bytes

For bf16, read raw bytes as `numpy.uint16`, then reinterpret via tinygrad:
```python
u16 = np.frombuffer(raw_bytes, dtype=np.uint16).copy()
t = Tensor(u16).bitcast(dtypes.bfloat16).reshape(shape)
```

Full implementation: `tinygrad_ft/hf_load.py::_load_safetensors_file`.

### HF ↔ tinygrad parameter naming table

Tinygrad's `Transformer` class exposes these paths via `nn.state.get_state_dict`:

| HuggingFace name                              | tinygrad name              |
|-----------------------------------------------|----------------------------|
| `model.embed_tokens.weight`                   | `token_embd.weight`        |
| `model.layers.<N>.input_layernorm.weight`     | `blk.<N>.attn_norm.weight` |
| `model.layers.<N>.self_attn.q_proj.weight`    | `blk.<N>.attn_q.weight`    |
| `model.layers.<N>.self_attn.k_proj.weight`    | `blk.<N>.attn_k.weight`    |
| `model.layers.<N>.self_attn.v_proj.weight`    | `blk.<N>.attn_v.weight`    |
| `model.layers.<N>.self_attn.o_proj.weight`    | `blk.<N>.attn_output.weight` |
| `model.layers.<N>.self_attn.q_norm.weight`    | `blk.<N>.attn_q_norm.weight` (Qwen3) |
| `model.layers.<N>.self_attn.k_norm.weight`    | `blk.<N>.attn_k_norm.weight` (Qwen3) |
| `model.layers.<N>.post_attention_layernorm.weight` | `blk.<N>.ffn_norm.weight` |
| `model.layers.<N>.mlp.gate_proj.weight`       | `blk.<N>.ffn_gate.weight`  |
| `model.layers.<N>.mlp.up_proj.weight`         | `blk.<N>.ffn_up.weight`    |
| `model.layers.<N>.mlp.down_proj.weight`       | `blk.<N>.ffn_down.weight`  |
| `model.norm.weight`                           | `output_norm.weight`       |
| `lm_head.weight`                              | `output.weight`            |

For models with **tied embeddings** (Llama 3.2, some Qwens), `lm_head.weight`
is absent — copy `token_embd.weight` to `output.weight` after loading:

```python
if "output.weight" not in state_dict and "token_embd.weight" in state_dict:
    state_dict["output.weight"] = state_dict["token_embd.weight"]
```

### Verifying correctness

After loading, ALL three should be true:

1. `len(state_dict) == num_model_params` (e.g., 311 for Qwen3-0.6B)
2. `handle.unmapped_keys() == []` (no orphan HF names)
3. `nn_state.load_state_dict(model, state_dict, strict=True)` passes

If `strict=True` load passes, your name mapping is correct.

---

## Building a training pipeline (the canonical shape)

Here's the end-to-end pattern that actually works. Every piece matters.

```python
from tinygrad import Tensor
from tinygrad.nn.optim import AdamW
from tinygrad_ft import (
    load_hf_model, build_model, apply_lora, get_lora_parameters,
    prepare_for_training, HFTokenizer, tokenize_batch, compute_loss,
)

# 1. Load HF weights into tinygrad (with bf16 bitcast + name remap)
handle = load_hf_model("Qwen/Qwen3-0.6B")

# 2. Instantiate Transformer + load weights strict
model = build_model(handle)

# 3. Apply LoRA. freeze_non_lora=True freezes all 311 base params; adds
#    224 new trainable tensors (112 adapters × A, B). Critical for Trap 5.
adapters = apply_lora(model, rank=8, alpha=16)

# 4. Swap each block's _attention with a cache-free training version.
#    Fixes Traps 2 and 3 (autograd through STORE + @function decorator).
prepare_for_training(model)

# 5. Get a tokenized batch. Pad positions get loss_mask=0 → ignored
#    via ignore_index=-1 in cross-entropy.
tokenizer = HFTokenizer(handle.model_path)
batch = tokenize_batch(examples, tokenizer, max_length=128)

# 6. Optimizer operates over just the LoRA params (A and B tensors).
#    AdamW state is ~10× the param count in float32.
optimizer = AdamW(get_lora_parameters(adapters), lr=1e-3)

# 7. Training step. Note the chained .backward() + realize pattern.
Tensor.training = True
for step in range(num_steps):
    optimizer.zero_grad()
    loss = compute_loss(model, batch).backward()  # schedules backward
    Tensor.realize(loss, *optimizer.schedule_step())  # realizes everything
    print(f"step {step} loss {float(loss.item()):.4f}")
```

### Cross-entropy loss with pad masking

```python
input_ids = batch["input_ids"]          # (B, T)
loss_mask = batch["loss_mask"]          # (B, T), 1=real, 0=pad
inputs  = input_ids[:, :-1]             # (B, T-1), for next-token prediction
targets = input_ids[:, 1:]
mask    = loss_mask[:, 1:]

logits = get_logits_train(model, inputs)  # (B, T-1, V)

# Set padded targets to -1 so sparse_categorical_crossentropy skips them
ignored_targets = (targets * mask) + (-1 * (1 - mask))

loss = logits.sparse_categorical_crossentropy(
    ignored_targets, ignore_index=-1, reduction="mean",
)
```

---

## LoRA adapter pattern (correct implementation)

```python
from tinygrad import Tensor
from tinygrad.nn import Linear

class LoRALinear:
    def __init__(self, base: Linear, rank: int = 8, alpha: int = 16):
        self.base = base
        self.base.weight.requires_grad = False  # freeze base
        if getattr(self.base, "bias", None) is not None:
            self.base.bias.requires_grad = False

        out_features, in_features = base.weight.shape
        bound = 1.0 / (in_features ** 0.5)

        # CRITICAL: requires_grad=True at construction, not post-hoc
        # (Trap 4). No .contiguous() before this (Trap 4).
        self.A = Tensor.uniform(rank, in_features, low=-bound, high=bound,
                                 requires_grad=True)
        self.B = Tensor.zeros(out_features, rank, requires_grad=True)

        self.rank = rank
        self.scale = alpha / rank

    def __call__(self, x: Tensor) -> Tensor:
        # y = W·x + (α/r)·B·A·x
        # Compute (x @ A.T) first → (..., r) intermediate, then @ B.T → (..., out)
        # Avoids materializing the full B·A matrix (defeats memory savings).
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scale

    def merge(self) -> Tensor:
        # For deployment: fold A and B into base weights.
        return self.base.weight + (self.B @ self.A) * self.scale
```

**Why B = zeros at init**: ensures the adapter is an identity function at
step 0 (`B·A·x == 0`). Training doesn't have to first "recover" the
pretrained behavior — it starts from the base and learns only
improvements. Removing this (e.g., random init on B) causes loss to
start much higher and training to be less stable.

---

## Performance tuning

### Environment variables that matter

| Var | Effect | Recommendation |
|---|---|---|
| `DEV` | Device selection (AMD, NV, METAL, CPU, ...) | Set at process start, always |
| `JITBEAM` | Beam search for JIT'd kernels | `2` for normal use, `4-8` for ultimate |
| `BEAM` | Global beam search (applied non-JIT too) | Prefer `JITBEAM` for LLMs |
| `PARALLEL` | Parallel kernel compilation during beam search | `10` on M4 Pro (P-core count) |
| `DEBUG` | Verbosity (1=ops, 2=timings, 4=gen code, 7=asm) | `2` is useful for timing issues |
| `CACHEDB` | Override kernel cache path | Default is fine |
| `JIT` | JIT level (0=off, 1=on, 2=on-no-graph) | Default `1` for training, test `0` if graph bugs |
| `HALF` | Use fp16 weights in `tinygrad.llm` | Default `1`, leave it |
| `REALIZE` | Realize during GGUF load | Default `0`, don't touch |

### JITBEAM cache behavior

- First run with `JITBEAM=2` on a new model: expect 15-30 min of kernel
  search before actual inference/training starts. Each unique kernel
  shape gets autotuned.
- Kernels are cached at `~/Library/Caches/tinygrad/cache.db` keyed on
  kernel hash. Subsequent runs skip warmup entirely.
- Cache is per-kernel-shape, so batch-size changes or context-length
  changes trigger new kernels to be compiled.
- **Don't delete `~/Library/Caches/tinygrad/`** unless you want to redo
  tuning. The database can grow to ~1-2 GB but it's all useful.

---

## Debugging recipes

### "My param's `.grad` is None"

1. Is `requires_grad=True` set at construction, not post-hoc? (Trap 4)
2. Did you freeze all non-optimizer params? (Trap 5)
3. Did you chain `.backward()` with `realize(*schedule_step())`? (Trap 1)
4. Is the param's `.uop` in `loss.uop.toposort()`? If not, graph was severed
   somewhere — bisect forward (Trap 2, 3).
5. Is `block.__call__` in your path? That has the `@function` decorator
   (Trap 2). Inline the block forward instead.

### "Training crashes with RuntimeError: failed to compute gradient for Ops.STORE"

You're using `tinygrad.llm.model.Transformer`'s stock attention. The
`cache_kv` STORE can't be backward'd. Call `prepare_for_training(model)`
before training (Trap 3).

### "loss.backward() raises AttributeError: 'int' object has no attribute 'item'"

`Tensor.numel()` in tinygrad returns a Python `int`, not a 0-dim Tensor.
Drop the `.item()`:

```python
count = int(tensor.numel())        # ✓
count = tensor.numel().item()      # ✗
```

### "SIGBUS on first forward pass"

If on macOS: You're probably relying on `Device.DEFAULT=METAL` and
calling `.to('AMD')`. Cross-device transfers on macOS 26 trigger libamd_comgr
crashes. Fix: `DEV=AMD python3 ...` at process start.

### "Loss starts at 11-12 and doesn't move"

Starting loss of `-log(1/V) ≈ log(vocab_size)` means the model is
effectively random — your weight loading is broken. Check:

- `handle.unmapped_keys()` returns `[]`
- `nn.state.load_state_dict(model, state_dict, strict=True)` passed
- Tied-embedding models (Llama 3.2) need `output.weight = token_embd.weight`

### "Loss goes down then explodes to NaN or bounces back up"

Usually learning rate too high. For LoRA on a small dataset:
- `lr=5e-3`: aggressive, can diverge on novel facts after ~10-15 steps
- `lr=1e-3`: stable baseline for overfit tests
- `lr=5e-4`: conservative for real training
- `lr=1e-4 to 5e-5`: multi-epoch instruction tuning

### "I get progress bars that look like URL spam"

Cosmetic stderr issue on macOS. Redirect: `2>/dev/null` or `2>/tmp/log`.
Harmless.

---

## Key classes and files in tinygrad you'll reference repeatedly

| File | What's there |
|---|---|
| `tinygrad/tensor.py` | `Tensor` class — the whole user-facing API |
| `tinygrad/nn/state.py` | `get_state_dict`, `get_parameters`, `load_state_dict`. Your primary state-loading tools. |
| `tinygrad/nn/optim.py` | `Optimizer`, `Adam`, `AdamW`, `SGD`, `LAMB`. Short, readable. |
| `tinygrad/nn/__init__.py` | `Linear`, `Embedding`, `RMSNorm`, `LayerNorm`. Simple. |
| `tinygrad/llm/model.py` | `Transformer`, `TransformerBlock`, `FFNBlock`, `TransformerConfig`. **This is where the four silent failures hide.** |
| `tinygrad/llm/cli.py` | Model registry for `tinygrad.llm` presets. Where `qwen3:30b-a3b` etc. are defined. |
| `tinygrad/engine/jit.py` | `TinyJit` + the `function` decorator. Read when debugging graph severance. |
| `tinygrad/nn/datasets.py` | MNIST et al. loaders. |
| `examples/beautiful_mnist.py` | **The authoritative training idiom.** When in doubt about training pattern, cargo-cult from here. |
| `examples/hlb_cifar10.py` | Speedrun-optimized CIFAR. More advanced patterns. |

---

## Library architecture pattern (from `tinygrad-ft`)

When building any tinygrad-based training library, this module split works:

```
your_package/
├── hf_load.py      # download + parse safetensors + name-map
├── build.py        # instantiate tinygrad model, load state_dict strict
├── forward.py      # inference-safe + training-safe forward passes
├── lora.py         # adapter primitives (if doing PEFT)
├── tokenizer.py    # thin wrapper around `tokenizers` lib
├── data.py         # tokenize + pad + loss mask → tensor batches
├── train.py        # compute_loss, train_step, overfit helpers
├── save.py         # save/load adapter weights (safetensors)
└── export_gguf.py  # (optional) merge + export for deployment
```

Why this split:
- `hf_load` is pure I/O; easy to test with unit tests on name mapping
- `build` is trivial once loader is right
- `forward` has the two variants because inference and training have
  different constraints (KV cache, JIT, sampling)
- `lora`/`data`/`train` are orthogonal concerns

If you're building something that doesn't involve LoRA (e.g., a bio
library), drop `lora.py` and `save.py` but keep the rest.

---

## Architecture support checklist (for adding a new model family)

To add Llama, Mistral, Gemma, etc., you'll need:

1. **Name mapping extensions in `hf_load.py::_map_hf_name_to_tinygrad`.**
   Check HF's `modeling_<family>.py` for weight names. Most families use
   Qwen/Llama conventions with minor tweaks.

2. **`SUPPORTED_ARCHITECTURES` entry.** Add the HF architecture string
   (e.g., `"LlamaForCausalLM"`).

3. **`_hf_config_to_tinygrad` translation.** Map HF's `config.json` fields
   to `TransformerConfig`. Watch for:
   - `num_key_value_heads` (GQA) may be absent for MHA models
   - `head_dim` may not be set → compute from `hidden_size / num_attention_heads`
   - `qk_norm` is Qwen3-specific; most families don't have it
   - `rope_theta` default varies by family

4. **RoPE weight permutation.** Some families (Llama) use interleaved RoPE
   layout in their safetensors; tinygrad expects half-split. Check
   `Transformer.from_gguf` for the reference permutation logic:
   ```python
   # For Llama-family attn_q.weight and attn_k.weight:
   w = weight.reshape(n_heads, weight.shape[0]//n_heads, -1)
   weight = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
   ```

5. **Tied embeddings.** If `lm_head.weight` is absent in the HF state dict,
   copy `token_embd.weight` to `output.weight` (already handled in
   `build_model`).

6. **Unit tests** for the name mapping. Add to `tests/test_name_mapping.py`.

7. **Integration test** loading a small model from the new family.

---

## Known limitations (as of this session)

1. **No PyTorch backend for tinygrad yet.** Upstream work exists but
   hasn't landed. When it does, PyTorch-based libraries (scVI, most HF
   Transformers code) will run on tinygrad backends with a 1-line change.

2. **MoE fine-tuning not tested** in `tinygrad-ft`. The `apply_lora`
   function skips `ExpertWeights` modules because they're not `nn.Linear`.
   MoE architectures would need a different adapter class.

3. **Only Qwen 2 / 3 / 3.5 dense** are tested. Llama, Mistral, Gemma are
   planned but require name mapping and possibly RoPE permutation work.

4. **No GGUF writer.** We can LOAD GGUF via tinygrad's stock code; we
   can't WRITE GGUF to deploy a fine-tuned model. Planned for `export_gguf.py`.

5. **macOS + AMD is the only supported stack in this guide.** NVIDIA on
   Linux should work identically but hasn't been tested in this session.
   Metal on Apple Silicon probably works for smaller models but no
   specific testing.

6. **No support for bias in LoRALinear yet.** Most attention projections
   have `bias=False` so this is rarely relevant, but MLP layers in some
   families do have biases.

7. **No gradient checkpointing.** For models >7B on our 20 GB VRAM,
   activations become the bottleneck. Not yet implemented.

---

## When debugging goes off the rails — escape hatches

If you're fully stuck after 30 minutes:

1. **Add a minimal reproducer that works from scratch** — not in the
   context of your full pipeline. Often reveals the issue immediately.

2. **Compare to `beautiful_mnist.py`.** If your training loop doesn't
   structurally match that one, something's probably wrong.

3. **Check the compute graph explicitly:**
   ```python
   all_uops = loss.uop.toposort()
   for p in get_lora_parameters(adapters):
       in_graph = p.uop in all_uops
       print(f"{p.shape}: in_graph={in_graph}")
   ```
   If `False`, your param isn't being used in the forward — look for
   `@function` decorators, `.contiguous()` seals, or stale module references.

4. **Compare to what `tinygrad.llm` does** for inference. That path is
   battle-tested — if your forward disagrees with `Transformer.forward`
   in structure, investigate the difference.

5. **Ask the tinygrad Discord** with a minimal repro. The team is small
   and responsive. They've seen most issues.

6. **Downgrade to Python 3.13.** If on 3.14, some C-extension ABI
   mismatches can produce weird SIGBUS. Not consistently reproducible
   but cheap to rule out.

---

## Reference: what already exists in `tinygrad-ft` that you can reuse

When building new functionality, check if these already exist (in
`tinygrad_ft/`):

| Need | Use |
|---|---|
| Download + load HF model | `load_hf_model("org/name")` returns `HFModelHandle` |
| Instantiate tinygrad Transformer | `build_model(handle)` |
| Get inference logits | `get_logits(model, tokens)` (doesn't work after prepare_for_training) |
| Get training logits | `get_logits_train(model, tokens)` (autograd-safe) |
| Monkey-patch model for training | `prepare_for_training(model)` |
| Add LoRA | `apply_lora(model, rank=r, alpha=a)` |
| List LoRA params | `get_lora_parameters(adapters)` |
| Count LoRA params | `count_lora_parameters(adapters)` |
| Tokenize a batch | `tokenize_batch(examples, tokenizer, max_length=N)` |
| Compute NTP loss | `compute_loss(model, batch)` (in `tinygrad_ft.train`) |
| Run overfit sanity check | `overfit(model, batch, optimizer, steps=N)` |

Before writing anything new, grep the library — most of this is there.

---

## Session origin

This guide comes from a single build session on 2026-04-18 that went from
"just set up the eGPU" to "working LoRA fine-tune + benchmarks + published
OSS repo" in one sitting. The bugs documented here are the ones that
actually cost us hours. See [`BUGS.md`](./BUGS.md) for the full post-mortem
with symptom/cause/fix detail on all 13 issues.

If this guide saves you time, that's a sufficient return on the
investment. If you find a new pitfall not listed here, please append it
— the document is deliberately a working log, not a finished spec.
