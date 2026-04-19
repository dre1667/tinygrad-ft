"""tinygrad-ft: LoRA fine-tuning and HuggingFace weight loading for tinygrad."""

__version__ = "0.0.1"

from .hf_load import load_hf_model, HFModelHandle
from .build import build_model
from .forward import get_logits, get_logits_train, prepare_for_training
from .lora import (
    LoRALinear,
    apply_lora,
    get_lora_parameters,
    count_lora_parameters,
    DEFAULT_LORA_TARGETS,
)
from .tokenizer import HFTokenizer
from .data import load_jsonl, tokenize_batch, TokenizedBatch
from .train import compute_loss, train_step, overfit, StepResult

__all__ = [
    # hf load
    "load_hf_model",
    "HFModelHandle",
    # model
    "build_model",
    "get_logits",
    "get_logits_train",
    "prepare_for_training",
    # lora
    "LoRALinear",
    "apply_lora",
    "get_lora_parameters",
    "count_lora_parameters",
    "DEFAULT_LORA_TARGETS",
    # tokenizer + data
    "HFTokenizer",
    "load_jsonl",
    "tokenize_batch",
    "TokenizedBatch",
    # training
    "compute_loss",
    "train_step",
    "overfit",
    "StepResult",
    "__version__",
]
