# tokenizer/train_tokenizer.py
#
# Trains a BPE tokenizer on Python code using a 3-source data mix:
#
#   Source 1 — codeparrot/github-code (Python, MIT/Apache licensed)
#              No login required. Millions of real GitHub Python files.
#              Used for broad coverage of Python syntax/patterns.
#
#   Source 2 — bigcode/the-stack-smol (Python subset, requires HF login)
#              10,000 high-quality deduplicated Python files.
#              Adds cleaner, more curated code to the mix.
#
#   Source 3 — Magicoder-OSS-Instruct-75K (instruction + solution pairs)
#              No login required. Adds docstrings, comments, typed code.
#              Makes the tokenizer better at natural language in code.
#
# This 3-source mix gives a tokenizer that handles:
#   - Raw scripts and library code (Source 1)
#   - Clean, deduplicated production code (Source 2)
#   - Documented, instruction-following code (Source 3)
#
# Runtime: ~10-15 minutes
# Run once, reuse forever.

import os
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from huggingface_hub import HfFolder
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
    processors,
    decoders,
)

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
VOCAB_SIZE = 32000
SAVE_DIR = Path("tokenizer/vocab")

# How many files to pull from each source
N_GITHUB_CODE = 150_000   # from codeparrot/github-code (Python)
N_STACK_SMOL = 10_000    # from bigcode/the-stack-smol  (all of it)
N_MAGICODER = 75_000    # from Magicoder OSS-Instruct  (all of it)

SPECIAL_TOKENS = [
    "<|endoftext|>",    # document separator / EOS
    "<|fim_prefix|>",   # FIM: prefix context
    "<|fim_suffix|>",   # FIM: suffix context
    "<|fim_middle|>",   # FIM: hole to fill
    "<|pad|>",          # padding
]


# ------------------------------------------------------------------ #
# Source 1 — codeparrot/github-code (no login needed)
# ------------------------------------------------------------------ #
def iter_github_code(n: int = N_GITHUB_CODE) -> Iterator[str]:
    print(f"\n[Source 1] codeparrot/github-code → up to {n:,} Python files")
    ds = load_dataset(
        "codeparrot/github-code",
        streaming=True,
        split="train",
        trust_remote_code=True,
    )
    ds = ds.filter(lambda x: x["language"] == "Python")

    count = 0
    for sample in ds:
        code = sample.get("code", "")
        if code and len(code) >= 100:
            yield code
            count += 1
            if count % 25_000 == 0:
                print(f"  Source 1: {count:,} files processed...")
            if count >= n:
                break
    print(f"  Source 1 done: {count:,} files")


# ------------------------------------------------------------------ #
# Source 2 — bigcode/the-stack-smol Python subset (needs HF login)
# ------------------------------------------------------------------ #
def iter_stack_smol() -> Iterator[str]:
    token = HfFolder.get_token()
    if not token:
        print("\n[Source 2] Skipping the-stack-smol — no HF token found.")
        print("  Run: python -c \"from huggingface_hub import HfFolder; HfFolder.save_token('YOUR_TOKEN')\"")
        return

    print(
        f"\n[Source 2] bigcode/the-stack-smol → Python subset ({N_STACK_SMOL:,} files)")
    try:
        ds = load_dataset(
            "bigcode/the-stack-smol",
            data_dir="data/python",
            split="train",
            token=token,
            trust_remote_code=True,
        )
        count = 0
        for sample in ds:
            code = sample.get("content", "")
            if code and len(code) >= 100:
                yield code
                count += 1
        print(f"  Source 2 done: {count:,} files")
    except Exception as e:
        print(f"  Source 2 skipped ({e})")


# ------------------------------------------------------------------ #
# Source 3 — Magicoder OSS-Instruct (no login needed)
# Adds typed, documented, instruction-following code patterns
# ------------------------------------------------------------------ #
def iter_magicoder() -> Iterator[str]:
    print(
        f"\n[Source 3] Magicoder-OSS-Instruct-75K → {N_MAGICODER:,} examples")
    try:
        ds = load_dataset(
            "ise-uiuc/Magicoder-OSS-Instruct-75K",
            split="train",
            trust_remote_code=True,
        )
        count = 0
        for sample in ds:
            # Each sample has 'problem' (docstring/instruction) + 'solution' (code)
            problem = sample.get("problem", "")
            solution = sample.get("solution", "")
            # Combine as: docstring + code (realistic Python file structure)
            combined = f'"""\n{problem}\n"""\n\n{solution}'
            if len(combined) >= 100:
                yield combined
                count += 1
                if count >= N_MAGICODER:
                    break
        print(f"  Source 3 done: {count:,} examples")
    except Exception as e:
        print(f"  Source 3 skipped ({e})")


# ------------------------------------------------------------------ #
# Combined iterator — yields from all 3 sources in sequence
# ------------------------------------------------------------------ #
def combined_python_iterator() -> Iterator[str]:
    """
    Yields Python code strings from all 3 sources.
    The BPE trainer sees a diverse mix of:
      - Raw GitHub scripts (broad syntax coverage)
      - Deduplicated production code (quality)
      - Instruction+solution pairs (docstrings, type hints)
    """
    yield from iter_github_code()
    yield from iter_stack_smol()
    yield from iter_magicoder()


# ------------------------------------------------------------------ #
# Train
# ------------------------------------------------------------------ #
def train_tokenizer():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    token = HfFolder.get_token()

    print("=" * 60)
    print("PyCraft-1 Tokenizer Training  (3-source data mix)")
    print("=" * 60)
    print(f"  Vocab size    : {VOCAB_SIZE:,}")
    print(
        f"  HF token      : {'found ✓' if token else 'not found — Source 2 will be skipped'}")
    print(f"  Save dir      : {SAVE_DIR}")
    print(f"  Special tokens: {SPECIAL_TOKENS}")

    # BPE tokenizer — byte-level, same as GPT-2 / Llama / StarCoder
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print("\nStarting BPE training across all sources...")
    tokenizer.train_from_iterator(
        combined_python_iterator(),
        trainer=trainer,
    )

    # Save
    tokenizer_path = SAVE_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"\nTokenizer saved → {tokenizer_path}")

    # Report special token IDs
    print("\nSpecial token IDs:")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok:<22} → {tokenizer.token_to_id(tok)}")

    # Sanity round-trip test
    print("\nSanity tests:")
    tests = [
        'def fibonacci(n: int) -> int:\n    """Return nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n',
        'import numpy as np\nfrom typing import List\n\ndef matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:\n    return np.dot(a, b)\n',
        'class DataLoader:\n    def __init__(self, batch_size: int = 32):\n        self.batch_size = batch_size\n',
    ]
    all_ok = True
    for i, code in enumerate(tests, 1):
        enc = tokenizer.encode(code)
        dec = tokenizer.decode(enc.ids)
        ok = dec.replace(" ", "") == code.replace(" ", "")
        print(
            f"  Test {i}: {len(enc.ids)} tokens — {'OK' if ok else 'MISMATCH'}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll round-trip tests passed.")
    else:
        print("\nWARNING: Some decode mismatches. Check tokenizer config.")

    # Token efficiency report
    sample = "def hello(name: str) -> str:\n    return f'Hello, {name}!'\n"
    enc = tokenizer.encode(sample)
    print(f"\nToken efficiency:")
    print(f"  '{sample.strip()}'")
    print(f"  → {len(enc.ids)} tokens (lower is better for code)")

    print("\n" + "=" * 60)
    print("Tokenizer training complete.")
    print("=" * 60)
    return tokenizer


if __name__ == "__main__":
    train_tokenizer()
