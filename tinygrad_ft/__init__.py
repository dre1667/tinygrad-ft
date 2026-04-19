"""tinygrad-ft: LoRA fine-tuning and HuggingFace weight loading for tinygrad."""

__version__ = "0.0.1"

from .hf_load import load_hf_model, HFModelHandle
from .build import build_model
from .forward import get_logits
from .lora import (
    LoRALinear,
    apply_lora,
    get_lora_parameters,
    count_lora_parameters,
    DEFAULT_LORA_TARGETS,
)

__all__ = [
    "load_hf_model",
    "HFModelHandle",
    "build_model",
    "get_logits",
    "LoRALinear",
    "apply_lora",
    "get_lora_parameters",
    "count_lora_parameters",
    "DEFAULT_LORA_TARGETS",
    "__version__",
]
