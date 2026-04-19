# Debug log — how `tinygrad-ft` came into existence

This is a working log of every non-obvious bug we hit while building this
library in a single session. It exists because:

1. If you hit one of these you'll save hours not re-debugging it
2. Several are undocumented silent failures that have bitten others and will
   bite others again
3. It makes the repo legitimate — nothing here is imaginary, every workaround
   in the source has a specific reason

Bugs are grouped by the layer they live in: (A) eGPU / macOS setup, (B) HF
loading into tinygrad, (C) tinygrad's autograd itself. Each entry has:
**symptom**, **cause**, **fix**.

---

## A. eGPU / macOS setup bugs

These bit us before we wrote a single line of `tinygrad-ft` code.

### A.1. Apple Silicon blocks user-space driver extensions by default

**Symptom.** The TinyGPU driver extension installs into `/Applications/TinyGPU.app`
and `systemextensionsctl list` shows it as `[activated waiting for user]`, but
the toggle in System Settings → Login Items & Extensions → Driver Extensions
is disabled and clicking it does nothing.

**Cause.** On Apple Silicon, DriverKit system extensions that need raw PCIe
+ DMA access (like eGPU drivers) are gated behind **Reduced Security** mode,
regardless of Apple Developer ID notarization. Notarization is necessary
but not sufficient.

**Fix.** Boot into Recovery Mode (hold power button on shutdown Mac),
open Startup Security Utility, set the boot volume to **Reduced Security**
AND check "Allow user management of kernel extensions from identified
developers." Reboot, then the toggle works.

---

### A.2. `Python 3.14 + libamd_comgr` SIGBUS looked like an ABI issue — it wasn't

**Symptom.** Inside tinygrad, the first compute kernel call crashes Python
with SIGBUS (`EXC_ARM_DA_ALIGN`). Crash report shows the fault deep inside
`libamd_comgr.dylib` → `clang::ExecuteCompilerInvocation` →
`llvm::cl::ParseCommandLineOptions`. Bad PC address looked like a PAC-signed
pointer that failed to authenticate:

```
EXC_BAD_ACCESS (SIGBUS)
EXC_ARM_DA_ALIGN at 0x37000200aa0003f3
  └─ llvm::cl::opt<bool>::handleOccurrence
  └─ amd_comgr_do_action
```

**Initial (wrong) diagnosis.** `libamd_comgr.dylib` was built against macOS
SDK 15.5 with older PAC-signing ABI, and macOS 26.3 tightened PAC enforcement,
so any virtual dispatch through the library's LLVM vtables fails.

We rebuilt the venv with Python 3.13 to eliminate the variable. **Still
crashed.** So PAC ABI wasn't the root cause.

**Actual cause.** We were running `Tensor([1,2,3]).to('AMD')` — setting
`Device.DEFAULT = "METAL"` (because we hadn't exported `DEV=AMD`) and then
trying to transfer to AMD. On macOS 26 the Metal → AMD device transfer path
in tinygrad triggers the libamd_comgr compiler in a way that crashes.

**Fix.** Set `DEV=AMD` as an environment variable *before* Python starts, so
tinygrad's `Device.DEFAULT` is `AMD` from the beginning. No cross-device
transfer, no crash. Simply:

```bash
DEV=AMD python3 -m tinygrad.llm --model qwen3:0.6b
```

**Lesson.** Crash reports that *look* like ABI issues can actually be
workload-path issues. Eliminating Python versions is a cheap test; do it
before investing in rebuilding system libraries.

---

### A.3. Tinygrad's progress bars render as URL spam on macOS terminals

**Symptom.** Running any `tinygrad.llm` command spews hundreds of lines like:

```
https://github.com/tinygrad/tinygpu_releases/raw/c0d024f9ff0e1dc8fdhttps://github.com/tinygrad/tinygpu_releases/raw/c0d024f9ff0e1dc8fdhttps://...
```

**Cause.** Tinygrad writes curl-style carriage-return-based progress bars to
stderr. Terminals that lack the TTY control sequences render these
back-to-back with no line breaks, producing the illusion that `curl` is
broken.

**Fix.** Redirect stderr: `2>/tmp/tg.log` or `2>/dev/null`. Purely cosmetic;
the commands work fine. Worth reporting upstream as a quality-of-life fix.

---

### A.4. `AMD=1` env var was silently renamed to `DEV=AMD`

**Symptom.** Old tutorials / Stack Overflow answers use `AMD=1 python …`
which errors with:

```
AssertionError: AMD=1 is deprecated, use DEV=AMD instead
```

**Cause.** tinygrad switched from per-device boolean env vars (`AMD=1`,
`METAL=1`, `NV=1`) to a unified `DEV=<backend>` system.

**Fix.** Use `DEV=AMD`, not `AMD=1`. Obvious once you see the error, but
many online references haven't been updated.

---

## B. Library bugs: HF safetensors → tinygrad

### B.1. numpy has no bfloat16 dtype, so `safetensors.safe_open(framework="numpy")` crashes on bf16 weights

**Symptom.**

```python
>>> from safetensors import safe_open
>>> with safe_open("model.safetensors", framework="numpy") as f:
...     arr = f.get_tensor("model.embed_tokens.weight")
TypeError: data type 'bfloat16' not understood
```

**Cause.** Modern HF models are bf16 by default. numpy doesn't support
bf16 natively. The `framework="numpy"` path in safetensors auto-converts,
which fails.

**Fix.** Parse safetensors manually: read 8-byte LE header size, parse the
JSON header, slice raw bytes for each tensor, and for bf16 read as uint16
and reinterpret via tinygrad's `bitcast` to `dtypes.bfloat16`. About 30
lines of code; lives in `hf_load.py::_load_safetensors_file`.

Avoids introducing a torch dependency (which would be the other common
path — safetensors + torch does handle bf16 correctly).

---

### B.2. HF ↔ tinygrad parameter naming mismatch, three places deep

**Symptom.** `nn.state.load_state_dict(model, state_dict, strict=True)`
complains about every parameter name.

**Cause.** HF and tinygrad use different conventions at every level:

| Level       | HuggingFace                            | tinygrad `Transformer`    |
|-------------|----------------------------------------|---------------------------|
| embed       | `model.embed_tokens.weight`            | `token_embd.weight`       |
| layers      | `model.layers.<N>.*`                   | `blk.<N>.*`               |
| final norm  | `model.norm.weight`                    | `output_norm.weight`      |
| attention   | `self_attn.q_proj / k_proj / v_proj / o_proj` | `attn_q / attn_k / attn_v / attn_output` |
| MLP         | `mlp.gate_proj / up_proj / down_proj`  | `ffn_gate / ffn_up / ffn_down` |
| norms       | `input_layernorm / post_attention_layernorm` | `attn_norm / ffn_norm` |
| head        | `lm_head.weight`                       | `output.weight`           |

There's no crosswalk in either project's docs. You have to read
`tinygrad/llm/model.py`'s `Transformer` class and match up against HF's
`modeling_qwen.py` or similar to see what each side calls each weight.

**Fix.** The `_map_hf_name_to_tinygrad()` function in `hf_load.py`.
Covered by `tests/test_name_mapping.py`.

---

### B.3. `tinygrad.Tensor.numel()` returns a Python `int`, not a Tensor

**Symptom.**

```python
total = sum(ad.A.numel().item() + ad.B.numel().item() for ad in adapters)
AttributeError: 'int' object has no attribute 'item'
```

**Cause.** Unlike PyTorch, tinygrad's `Tensor.numel()` returns a plain
`int`, not a 0-dim tensor. Calling `.item()` on it errors.

**Fix.** Drop the `.item()`: `int(ad.A.numel())`.

---

### B.4. `pip install tinygrad` doesn't include the `tinygrad.llm` submodule

**Symptom.** Fresh venv, `pip install tinygrad-ft` (which declares
`tinygrad>=0.12.0`), then `from tinygrad.llm.model import TransformerConfig`
fails with `ModuleNotFoundError`.

**Cause.** PyPI `tinygrad==0.12.0` was packaged before the `llm` subpackage
was added. The module only exists on master.

**Fix (for users).** `pip install git+https://github.com/tinygrad/tinygrad.git`
before installing tinygrad-ft. Documented in the README quickstart.

**Fix (upstream).** Tinygrad needs to cut a 0.12.1 or 0.13 release that
includes `llm/`. This is in their hands.

---

## C. The autograd bugs — four silent failures that don't raise

These are the reason the "training pipeline" step took four times longer
than "loading weights" and "running forward pass" combined. Each one
produces *no error* — gradients silently don't exist, or don't flow, and
the optimizer just assertions against `None`.

### C.1. `loss.backward()` alone does not populate `.grad`

**Symptom.** After:

```python
optimizer.zero_grad()
loss = compute_loss(model, batch)
loss.backward()
optimizer.step()  # AssertionError: unwrap(None)
```

every LoRA param has `.grad is None`, and the optimizer errors on the
first `unwrap`.

**Cause.** Tinygrad is lazy. `.backward()` schedules gradient
computations but doesn't run them. `.grad` is populated only when the
scheduled graph is *realized*.

**Fix.** Chain backward and the optimizer's schedule into a single
realize call. This is also what `tinygrad/examples/beautiful_mnist.py`
does:

```python
loss = compute_loss(model, batch).backward()
Tensor.realize(loss, *optimizer.schedule_step())
```

Now `.grad` materializes and the optimizer update happens in the same
pass.

---

### C.2. The stock attention's KV-cache `STORE` op can't be differentiated

**Symptom.** Training raises:

```
RuntimeError: failed to compute gradient for Ops.STORE
```

pointing at the attention layer's `cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(...)`.

**Cause.** `tinygrad/llm/model.py`'s `TransformerBlock._attention` writes
the current (k, v) into an in-place `cache_kv` tensor via an explicit
`STORE` UOp. That's fine for autoregressive generation (where the
cache is the whole point) but tinygrad's autograd has no gradient for
`STORE`.

**Fix.** Write a training-mode attention that skips the cache and uses
the current (k, v) directly — it's what we'd want anyway for a full-sequence
forward pass. Monkey-patch each block's `_attention` at training-prep
time. Source: `forward.py::_attention_train`, `forward.py::prepare_for_training`.

**Upstream opportunity.** Add a `training=True` path to the stock attention
that skips the KV write. Well-scoped PR.

---

### C.3. `@function(precompile=True, allow_implicit=True)` decorator silently severs autograd

**Symptom.** After fixing C.1 and C.2, `.grad` is *still* `None` on every
LoRA param. Loss computes correctly (numerically reasonable value), but
`A.uop not in loss.uop.toposort()`. The param isn't in the compute graph
somehow.

**Cause.** `FFNBlock.__call__` wraps its own body in
`@function(precompile=True, allow_implicit=True)`:

```python
def __call__(self, x, start_pos):
    self._init_state(x)
    @function(precompile=True, allow_implicit=True)
    def _run(x, start_pos):
        h = x + self._attention(self.attn_norm(x), start_pos)
        return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos)
```

That decorator captures `_run`'s compute graph on first call and treats
closure-referenced state (`self._attention`, `self.attn_norm`, etc.) as
**implicit constants**. The LoRA A and B tensors reached via `self` are
frozen into the captured graph as if they were constants — so they never
appear as differentiable inputs.

**Detection method.** Bisect by calling pieces of the block manually:

```python
attn = block._attention(block.attn_norm(x), 0)
# A.uop in attn.uop.toposort()? → True
block_out = block(x, 0)
# A.uop in block_out.uop.toposort()? → False
```

The `_attention` call alone preserves autograd. Wrapping it in
`block.__call__` destroys it. Finding the `@function` decorator in the
source at that point is obvious in hindsight.

**Fix.** Bypass `block.__call__` entirely in training. Inline the block's
forward manually:

```python
def get_logits_train(model, tokens):
    x = model.token_embd(tokens).float()
    for block in model.blk:
        block._init_state(x)
        h = x + block._attention(block.attn_norm(x), 0)
        x = (h + block._feed_forward(block.ffn_norm(h))).contiguous()
    return model.output(model.output_norm(x))
```

Source: `forward.py::get_logits_train`.

**Upstream opportunity.** Expose an un-decorated `_forward` method so
external training code doesn't have to reimplement the block body.

---

### C.4. `Tensor.uniform(...).requires_grad_(True)` doesn't register with the optimizer

**Symptom.** `get_parameters` finds your LoRA tensors with
`requires_grad=True`, forward runs through them, backward computes
logits — but after `Tensor.realize(*opt.schedule_step())`, the optimizer
says `t.grad is None`.

**Cause.** Tinygrad's factory functions (`Tensor.uniform`, `Tensor.zeros`,
`.contiguous()`) return tensors whose `.uop` is already a realized leaf.
Calling `.requires_grad_(True)` in place *after* that flips a flag, but
the leaf's identity inside the compute graph has already been "sealed"
in a way that autograd doesn't retroactively pick up.

**Fix.** Pass `requires_grad=True` at the factory call, not after:

```python
# ❌ silently broken
self.A = Tensor.uniform(rank, in_features).requires_grad_(True)

# ✅ correct
self.A = Tensor.uniform(rank, in_features, requires_grad=True)
```

Source: `lora.py::LoRALinear.__init__`.

---

### C.5 (bonus). Non-LoRA `requires_grad=True` tensors poison the backward pass

**Symptom.** Even after fixing C.1 through C.4, training still produced
`grad=None` on LoRA params. Loss evaluated correctly, params *were* in the
graph, but none had gradients.

**Cause.** Empirically, in tinygrad, if there are many tensors in the compute
graph with `requires_grad=True` but they're not actually being optimized
(because you only pass a subset to the optimizer), gradients on the
actually-optimized subset silently fail to populate.

We didn't track this down to a single line in tinygrad's source. Our
working theory is that tinygrad's backward assumes the `requires_grad=True`
set IS the training set, and behaves unpredictably when it isn't.

**Fix.** Explicitly freeze every model parameter before inserting LoRA
adapters. `apply_lora(model)` now does this by default via
`freeze_non_lora=True`:

```python
from tinygrad.nn import state as nn_state
for t in nn_state.get_parameters(model):
    t.requires_grad = False   # everything off
# … then create A and B with requires_grad=True (only trainable things)
```

After this, backward populates `.grad` on the LoRA params as expected.

**Upstream opportunity.** This is either a real bug or an unwritten
invariant. Worth filing a minimal repro against tinygrad's issues.

---

## Summary

We hit **11 distinct issues** getting from "eGPU powered on" to
"LoRA fine-tune working end-to-end":

| Layer | Issue | Severity | Documented upstream? |
|-------|-------|----------|----------------------|
| macOS setup | A.1 Reduced Security required | Process blocker | No |
| macOS setup | A.2 SIGBUS from cross-device transfer | Crash | No |
| macOS setup | A.3 Progress bars look like URL spam | Cosmetic | No |
| macOS setup | A.4 `AMD=1` deprecated to `DEV=AMD` | Friction | Error message |
| HF loading  | B.1 numpy can't represent bf16 | Crash | No |
| HF loading  | B.2 Name-mapping crosswalk missing | Blocker | No |
| HF loading  | B.3 `numel().item()` tinygrad quirk | Crash | No |
| HF loading  | B.4 PyPI tinygrad lacks `llm/` | Install blocker | No |
| autograd    | C.1 Lazy eval + `.grad=None` | Silent | No |
| autograd    | C.2 Cache STORE op blocks backward | Crash | No |
| autograd    | C.3 `@function` severs autograd | Silent | No |
| autograd    | C.4 `.requires_grad_(True)` post-hoc fails | Silent | No |
| autograd    | C.5 Non-trainable `requires_grad=True` poisons backward | Silent | No |

The autograd bugs (C.1–C.5) are the ones most worth reporting upstream.
Four of them produce **silent wrong behavior with no error** — the kind
of bug that eats developer-days.

---

## Benchmark: does the training actually work?

After getting all of the above right, `tinygrad-ft` can LoRA-fine-tune
Qwen3-0.6B on a 7900 XT eGPU in a few minutes. Representative run from
`examples/benchmark_finetune.py`:

```
BEFORE training
  "The capital of France is"     →  P(Paris) = 0.66
  "The capital of Germany is"    →  P(Berlin) = 0.48
  "The capital of Japan is"      →  P(Tokyo) = 0.40
  "Project Alpha uses port"      →  P(42)   = 8e-07      (model hasn't seen this)
  "Project Beta uses port"       →  P(17)   = 2e-05      (model hasn't seen this)

Training:  LoRA rank=8, lr=1e-3, 50 steps, 5 examples
loss: 4.398 → 0.143 (97% reduction)

AFTER training
  "The capital of France is"     →  P(Paris)  = 0.9997
  "The capital of Germany is"    →  P(Berlin) = 0.9998
  "The capital of Japan is"      →  P(Tokyo)  = 0.9997
  "Project Alpha uses port"      →  P(42)     = 1.0000   (learned!)
  "Project Beta uses port"       →  P(17)     = 1.0000   (learned!)
```

Known facts: moderate-confidence (40–66%) → near-certainty (>99.9%).
Invented facts: effectively zero → 100%. The training loop does real work.
