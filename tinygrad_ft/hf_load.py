"""Load HuggingFace safetensors models into tinygrad.

Bridges HF naming conventions (model.layers.{i}.self_attn.q_proj.weight)
to tinygrad's Transformer module names (attn_q, ffn_gate, etc.).

Currently supports: Qwen 3 / Qwen 3.5 dense models.
Planned: Llama 3.x, Qwen MoE variants.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download
from tinygrad import Tensor, dtypes
from tinygrad.llm.model import TransformerConfig


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

SUPPORTED_ARCHITECTURES = {"Qwen3ForCausalLM", "Qwen2ForCausalLM"}


def _hf_config_to_tinygrad(cfg: dict[str, Any]) -> TransformerConfig:
    """Translate a HuggingFace config.json dict to a tinygrad TransformerConfig.

    Not every HF field has an equivalent; we pick reasonable defaults for those
    that matter for inference/training of dense Qwen-family models.
    """
    arch = cfg.get("architectures", [""])[0]
    if arch not in SUPPORTED_ARCHITECTURES:
        raise NotImplementedError(
            f"Architecture {arch!r} not yet supported. Supported: {SUPPORTED_ARCHITECTURES}"
        )

    head_dim = cfg.get("head_dim") or (cfg["hidden_size"] // cfg["num_attention_heads"])
    # Qwen 3 uses QK-norm on the full head dim; Qwen 2 does not.
    qk_norm = head_dim if arch == "Qwen3ForCausalLM" else 0

    return TransformerConfig(
        num_blocks=cfg["num_hidden_layers"],
        dim=cfg["hidden_size"],
        hidden_dim=cfg["intermediate_size"],
        n_heads=cfg["num_attention_heads"],
        n_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        norm_eps=cfg.get("rms_norm_eps", 1e-6),
        vocab_size=cfg["vocab_size"],
        head_dim=head_dim,
        rope_theta=cfg.get("rope_theta", 1_000_000.0),
        rope_dim=head_dim,
        v_head_dim=head_dim,
        max_context=cfg.get("max_position_embeddings", 8192),
        qk_norm=qk_norm,
    )


# ---------------------------------------------------------------------------
# Name mapping: HF → tinygrad
# ---------------------------------------------------------------------------

def _map_hf_name_to_tinygrad(hf_name: str) -> str | None:
    """Translate a HuggingFace parameter name to the corresponding tinygrad
    attribute path. Returns None if the parameter should be skipped.

    Examples:
        model.embed_tokens.weight -> tok_embeddings.weight
        model.layers.5.self_attn.q_proj.weight -> layers.5.attn_q.weight
        model.layers.5.mlp.gate_proj.weight -> layers.5.ffn_gate.weight
        model.norm.weight -> norm.weight
        lm_head.weight -> output.weight
    """
    # strip leading "model." except for lm_head
    name = hf_name
    if name.startswith("model."):
        name = name[len("model."):]

    # top-level
    if name == "embed_tokens.weight":
        return "tok_embeddings.weight"
    if name == "norm.weight":
        return "norm.weight"
    if hf_name == "lm_head.weight":
        return "output.weight"

    # per-layer rewrites
    replacements = {
        "self_attn.q_proj":           "attn_q",
        "self_attn.k_proj":           "attn_k",
        "self_attn.v_proj":           "attn_v",
        "self_attn.o_proj":           "attn_output",
        "self_attn.q_norm":           "attn_q_norm",
        "self_attn.k_norm":           "attn_k_norm",
        "mlp.gate_proj":              "ffn_gate",
        "mlp.up_proj":                "ffn_up",
        "mlp.down_proj":              "ffn_down",
        "input_layernorm":            "attn_norm",
        "post_attention_layernorm":   "ffn_norm",
    }
    for hf_segment, tg_segment in replacements.items():
        if hf_segment in name:
            return name.replace(hf_segment, tg_segment)

    # unmapped — return as-is so the caller can log it
    return name


# ---------------------------------------------------------------------------
# Tensor loading
# ---------------------------------------------------------------------------

# safetensors numpy dtype mapping (what numpy *can* represent natively)
_NP_DTYPES = {
    "F64":  np.float64,
    "F32":  np.float32,
    "F16":  np.float16,
    "I64":  np.int64,
    "I32":  np.int32,
    "I16":  np.int16,
    "I8":   np.int8,
    "U8":   np.uint8,
    "BOOL": np.bool_,
}


def _load_safetensors_file(path: Path) -> dict[str, Tensor]:
    """Read a single .safetensors file into a {name: tinygrad.Tensor} dict.

    We parse safetensors manually instead of using safetensors.safe_open because
    numpy has no bfloat16 dtype — the standard library auto-converts on load and
    crashes. Parsing the header lets us read raw bytes for bf16 and reinterpret
    them as tinygrad bf16 via bitcast.

    Safetensors format: 8-byte LE uint64 header-size, then JSON header, then
    raw tensor bytes. See https://github.com/huggingface/safetensors.
    """
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size).decode("utf-8"))
        data = f.read()  # remainder = raw tensor storage

    out: dict[str, Tensor] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        dtype_str: str = meta["dtype"]
        shape: list[int] = meta["shape"]
        start, end = meta["data_offsets"]
        raw = data[start:end]

        if dtype_str == "BF16":
            # numpy can't hold bf16 — grab raw bytes as uint16 and let tinygrad
            # reinterpret.
            u16 = np.frombuffer(raw, dtype=np.uint16).copy()
            t = Tensor(u16).bitcast(dtypes.bfloat16).reshape(shape)
        elif dtype_str in _NP_DTYPES:
            arr = np.frombuffer(raw, dtype=_NP_DTYPES[dtype_str]).copy()
            # empty tensors (0-dim or size-0) need explicit reshape
            t = Tensor(arr).reshape(shape) if shape else Tensor(arr.item())
        else:
            raise NotImplementedError(f"unsupported safetensors dtype: {dtype_str!r}")

        out[key] = t
    return out


@dataclass
class HFModelHandle:
    """The parts we loaded from HF that you'll need downstream."""
    config: TransformerConfig
    hf_config: dict[str, Any]
    state_dict: dict[str, Tensor]  # keys remapped to tinygrad names
    model_path: Path               # local snapshot dir (tokenizer.json etc. live here)

    def unmapped_keys(self) -> list[str]:
        """Keys that we couldn't translate — useful for debugging new architectures."""
        return sorted(k for k in self.state_dict if k.startswith("_unmapped::"))


def load_hf_model(model_id: str, cache_dir: str | Path | None = None) -> HFModelHandle:
    """Download + load a HuggingFace LLM into tinygrad tensors.

    Args:
        model_id: HF model identifier, e.g. "Qwen/Qwen3-1.5B".
        cache_dir: optional local cache path; defaults to HF's standard cache.

    Returns:
        HFModelHandle with remapped state_dict and populated TransformerConfig.
    """
    # 1. fetch the snapshot (safetensors + config.json + tokenizer.json)
    local_dir = Path(
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "tokenizer.model",  # some Llama variants
            ],
        )
    )

    # 2. parse config.json → TransformerConfig
    cfg = json.loads((local_dir / "config.json").read_text())
    tg_config = _hf_config_to_tinygrad(cfg)

    # 3. load all safetensors shards, merge, and remap keys
    raw_state: dict[str, Tensor] = {}
    for st in sorted(local_dir.glob("*.safetensors")):
        raw_state.update(_load_safetensors_file(st))

    remapped: dict[str, Tensor] = {}
    for hf_key, tensor in raw_state.items():
        tg_key = _map_hf_name_to_tinygrad(hf_key)
        if tg_key is None:
            continue
        # A mapped key that starts with its own HF-path means we couldn't
        # translate it — surface it but don't drop it.
        if tg_key == hf_key and not hf_key.startswith(("layers.", "norm.", "tok_embeddings.", "output.")):
            remapped[f"_unmapped::{hf_key}"] = tensor
        else:
            remapped[tg_key] = tensor

    return HFModelHandle(
        config=tg_config,
        hf_config=cfg,
        state_dict=remapped,
        model_path=local_dir,
    )
