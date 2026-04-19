"""Unit tests for HF → tinygrad parameter name translation.

These tests don't download anything — they just verify the naming bridge
produces the exact module-path strings that tinygrad's `Transformer` class
exposes via `nn.state.get_state_dict`.
"""
from tinygrad_ft.hf_load import _map_hf_name_to_tinygrad


def test_embed_tokens():
    assert _map_hf_name_to_tinygrad("model.embed_tokens.weight") == "token_embd.weight"


def test_lm_head():
    assert _map_hf_name_to_tinygrad("lm_head.weight") == "output.weight"


def test_final_norm():
    assert _map_hf_name_to_tinygrad("model.norm.weight") == "output_norm.weight"


def test_attention_projections():
    assert _map_hf_name_to_tinygrad("model.layers.5.self_attn.q_proj.weight") == "blk.5.attn_q.weight"
    assert _map_hf_name_to_tinygrad("model.layers.5.self_attn.k_proj.weight") == "blk.5.attn_k.weight"
    assert _map_hf_name_to_tinygrad("model.layers.5.self_attn.v_proj.weight") == "blk.5.attn_v.weight"
    assert _map_hf_name_to_tinygrad("model.layers.5.self_attn.o_proj.weight") == "blk.5.attn_output.weight"


def test_mlp_projections():
    assert _map_hf_name_to_tinygrad("model.layers.0.mlp.gate_proj.weight") == "blk.0.ffn_gate.weight"
    assert _map_hf_name_to_tinygrad("model.layers.0.mlp.up_proj.weight")   == "blk.0.ffn_up.weight"
    assert _map_hf_name_to_tinygrad("model.layers.0.mlp.down_proj.weight") == "blk.0.ffn_down.weight"


def test_layernorms():
    assert _map_hf_name_to_tinygrad("model.layers.12.input_layernorm.weight") == "blk.12.attn_norm.weight"
    assert _map_hf_name_to_tinygrad("model.layers.12.post_attention_layernorm.weight") == "blk.12.ffn_norm.weight"


def test_qk_norm_qwen3():
    assert _map_hf_name_to_tinygrad("model.layers.3.self_attn.q_norm.weight") == "blk.3.attn_q_norm.weight"
    assert _map_hf_name_to_tinygrad("model.layers.3.self_attn.k_norm.weight") == "blk.3.attn_k_norm.weight"
