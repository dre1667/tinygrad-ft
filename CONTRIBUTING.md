# Contributing to tinygrad-ft

Thanks for considering a contribution. This project is young and rough around the edges, so any help is genuinely welcome.

## Ways to contribute

### 1. Add a new architecture

The fastest path is usually extending [`tinygrad_ft/hf_load.py`](./tinygrad_ft/hf_load.py):

- Add the HF architecture string to `SUPPORTED_ARCHITECTURES`
- Extend `_hf_config_to_tinygrad` to translate any architecture-specific fields
- Extend `_map_hf_name_to_tinygrad` with new name rewrites if needed
- Add unit tests in [`tests/test_name_mapping.py`](./tests/test_name_mapping.py)
- Run the end-to-end smoke test against a small model in that family and confirm `unmapped_keys()` returns `[]`

### 2. Benchmark on new hardware

Open an issue or PR with:

- CPU / GPU / platform
- Python + tinygrad versions
- A small reproducer (`load_hf_model("Qwen/Qwen3-0.6B")` is fine)
- Observed behavior / any errors

### 3. Fix bugs

Bug reports should include the minimal failing input, your environment (`pip freeze | grep -E "tinygrad|numpy|safetensors|huggingface"`), and the full traceback. PRs fixing the bug should include a regression test.

### 4. Documentation

The README and ROADMAP are the highest-signal docs. Improvements to either (typos, clarifications, better examples) are always welcome.

## Development setup

```bash
git clone https://github.com/dre1667/tinygrad-ft
cd tinygrad-ft
python3.13 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

## Code style

- Format with `ruff format` before committing
- Lint with `ruff check`
- Line length 120 (set in `pyproject.toml`)
- Type hints on public functions
- Docstrings explaining non-obvious choices (why, not what)

## Commit messages

Short imperative subject line, blank line, then a paragraph if needed. Reference issues with `#123`.

Good:
```
Add Llama 3 name mapping + regression test

Qwen and Llama share most naming but differ on rotary
embedding config — adds a branch in _hf_config_to_tinygrad
for Llama3ForCausalLM. Tested with Llama-3.2-1B-Instruct.
```

Bad:
```
updates
```

## Scope boundaries

Things this project will not take:

- PyTorch as a dependency (defeats the purpose)
- Framework wrappers over tinygrad's own primitives that hide what's happening
- Features that require specific cloud providers or hardware we can't test on locally

## Questions

Open an issue with the `question` label, or start a discussion.
