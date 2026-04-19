"""Minimal wrapper around HuggingFace's `tokenizers` library.

We don't need the full `transformers.AutoTokenizer` machinery (chat
templates, special-token handling, etc.) for training. The raw
`tokenizers.Tokenizer.from_file("tokenizer.json")` gives us encode/decode
and that's enough to compute next-token-prediction loss.

Every modern HF model ships a `tokenizer.json` in its snapshot, so the
loader in `hf_load.py` already pulled it alongside the weights.
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


class HFTokenizer:
    """Thin wrapper: encode strings to token-ID lists, decode back."""

    def __init__(self, model_path: str | Path):
        tokenizer_file = Path(model_path) / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found in {model_path}. "
                f"Did you call load_hf_model() first?"
            )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Text → list of int token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """List of token IDs → decoded string."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
