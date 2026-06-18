# scripts/download_pretrain_data.py — final version
# All sources are small single-file downloads (< 100MB each).
# No bulk parquet downloads. No connection drops.
# Total: ~500k curated Python examples.

import re
import time
import shutil
from pathlib import Path
from datasets import load_dataset, load_from_disk, concatenate_datasets
from huggingface_hub import HfFolder

SAVE_PATH = Path("data/pretrain_python")
PARTS_PATH = Path("data/pretrain_parts")


def is_python_code(code: str) -> bool:
    if not code or len(code) < 60:
        return False
    non_python = [
        "function(", "const ", "let ", "var ", "=>",
        "public static void", "#include", "fn main()",
        "package main", "using System", "<?php",
        "export default", "import React", "func ",
    ]
    for pat in non_python:
        if pat in code:
            return False
    python_patterns = [
        r'\bdef \w+\s*\(',
        r'\bclass \w+',
        r'\bimport \w+',
        r'\bfrom \w+ import',
        r'\bprint\s*\(',
        r'\breturn\b',
        r'\bif \w+',
    ]
    return any(re.search(p, code) for p in python_patterns)


def cached_or_download(part_name: str, loader_fn) -> tuple[str, int] | None:
    """Run loader_fn and cache result. Skip if already cached."""
    part = PARTS_PATH / part_name
    if part.exists():
        ds = load_from_disk(str(part))
        print(f"  [{part_name}] cached: {len(ds):,} rows")
        return part_name, len(ds)
    try:
        ds = loader_fn()
        if ds is None or len(ds) == 0:
            return None
        ds.save_to_disk(str(part))
        print(f"  [{part_name}] saved: {len(ds):,} rows")
        return part_name, len(ds)
    except Exception as e:
        print(f"  [{part_name}] FAILED: {type(e).__name__}: {str(e)[:100]}")
        return None


# ------------------------------------------------------------------ #
# Source A — tiny-codes (~120k Python educational examples, ~150MB)
# ------------------------------------------------------------------ #
def load_a():
    print("\n[A] nampdn-ai/tiny-codes — Phi-1 style educational Python")
    ds = load_dataset("nampdn-ai/tiny-codes", split="train",
                      trust_remote_code=True)
    ds = ds.filter(
        lambda x: str(x.get("programming_language", "")).lower() == "python",
        desc="  Filter Python")
    ds = ds.map(lambda x: {"code": x.get("response", "")},
                remove_columns=ds.column_names, desc="  Normalise")
    return ds.filter(lambda x: is_python_code(x["code"]),
                     desc="  Validate")


# ------------------------------------------------------------------ #
# Source B — Magicoder (~48k Python OSS instruction+solution, ~60MB)
# ------------------------------------------------------------------ #
def load_b():
    print("\n[B] Magicoder-OSS-Instruct-75K — real OSS Python solutions")
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K",
                      split="train", trust_remote_code=True)
    ds = ds.map(
        lambda x: {"code": f'"""\n{x["problem"]}\n"""\n\n{x["solution"]}'},
        remove_columns=ds.column_names, desc="  Build")
    return ds.filter(lambda x: is_python_code(x["code"]),
                     desc="  Validate")


# ------------------------------------------------------------------ #
# Source C — Alpaca Python (~18k curated pairs, ~5MB)
# ------------------------------------------------------------------ #
def load_c():
    print("\n[C] python_code_instructions_18k_alpaca — curated pairs")
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca",
                      split="train", trust_remote_code=True)

    def proc(x):
        ctx = f"# Input: {x['input']}\n" if x.get("input", "").strip() else ""
        return {"code": f"# Task: {x['instruction']}\n{ctx}\n{x['output']}"}
    ds = ds.map(proc, remove_columns=ds.column_names, desc="  Build")
    return ds.filter(lambda x: is_python_code(x["code"]),
                     desc="  Validate")


# ------------------------------------------------------------------ #
# Source D — the-stack-smol (~9k deduplicated Python, cached, ~87MB)
# ------------------------------------------------------------------ #
def load_d():
    print("\n[D] bigcode/the-stack-smol — deduplicated production Python")
    token = HfFolder.get_token()
    if not token:
        print("  SKIPPED: No HF token.")
        return None
    ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python",
                      split="train", token=token, trust_remote_code=True)
    ds = ds.map(lambda x: {"code": x.get("content", "")},
                remove_columns=ds.column_names, desc="  Normalise")
    return ds.filter(lambda x: is_python_code(x["code"]),
                     desc="  Validate")


# ------------------------------------------------------------------ #
# Source E — flytech/python-codes-25k (~25k, single 23MB file)
# ------------------------------------------------------------------ #
def load_e():
    print("\n[E] flytech/python-codes-25k — cleaned Python tasks")
    ds = load_dataset("flytech/python-codes-25k", split="train",
                      trust_remote_code=True)

    def proc(x):
        inst = x.get("instruction", "")
        inp = x.get("input", "")
        out = x.get("output", "")
        ctx = f"# Input: {inp}\n" if inp.strip() else ""
        return {"code": f"# Task: {inst}\n{ctx}\n{out}"}
    ds = ds.map(proc, remove_columns=ds.column_names, desc="  Build")
    return ds.filter(lambda x: is_python_code(x["code"]),
                     desc="  Validate")


# ------------------------------------------------------------------ #
# Source F — iamtarun/code_instructions_120k_alpaca
# Full 120k version (includes all languages, filter to Python)
# Single ~30MB download
# ------------------------------------------------------------------ #
def load_f():
    print("\n[F] code_instructions_120k_alpaca — 120k multi-lang, Python filter")
    ds = load_dataset("iamtarun/code_instructions_120k_alpaca",
                      split="train", trust_remote_code=True)

    def proc(x):
        inst = x.get("instruction", "")
        inp = x.get("input", "")
        out = x.get("output", "")
        ctx = f"# Input: {inp}\n" if inp.strip() else ""
        return {"code": f"# Task: {inst}\n{ctx}\n{out}"}
    ds = ds.map(proc, remove_columns=ds.column_names, desc="  Build")
    return ds.filter(lambda x: is_python_code(x["code"]),
                     desc="  Validate Python")


# ------------------------------------------------------------------ #
# Source G — sahil2801/code_instructions_120k (~120k, ~25MB)
# ------------------------------------------------------------------ #
def load_g():
    print("\n[G] sahil2801/code_instructions_120k — diverse code instructions")
    ds = load_dataset("sahil2801/code_instructions_120k",
                      split="train", trust_remote_code=True)

    def proc(x):
        inst = x.get("instruction", "")
        output = x.get("output", "")
        return {"code": f"# {inst}\n\n{output}"}
    ds = ds.map(proc, remove_columns=ds.column_names, desc="  Build")
    return ds.filter(lambda x: is_python_code(x["code"]),
                     desc="  Validate Python")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    if SAVE_PATH.exists():
        print(f"Dataset already exists at {SAVE_PATH}")
        print(f"Delete with:  rmdir /s /q data\\pretrain_python")
        return

    PARTS_PATH.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PyCraft-1 — Curated Python Dataset (7 small sources)")
    print("=" * 60)
    print("All sources are small single files — no connection drops.")
    print()

    t0 = time.time()
    sources = [
        ("part_a", load_a),
        ("part_b", load_b),
        ("part_c", load_c),
        ("part_d", load_d),
        ("part_e", load_e),
        ("part_f", load_f),
        ("part_g", load_g),
    ]

    parts = []
    labels = []
    for part_name, loader_fn in sources:
        result = cached_or_download(part_name, loader_fn)
        if result:
            name, count = result
            part_ds = load_from_disk(str(PARTS_PATH / part_name))
            parts.append(part_ds)
            labels.append((part_name, count))

    if not parts:
        print("\nERROR: All sources failed.")
        return

    print(f"\n{'='*60}")
    print("Combining and shuffling...")
    combined = concatenate_datasets(parts)
    combined = combined.shuffle(seed=42)
    combined = combined.filter(
        lambda x: 80 <= len(x["code"]) <= 8000,
        desc="Length filter (80-8000 chars)",
    )

    print(f"\nSource breakdown:")
    source_names = [
        "tiny-codes", "magicoder", "alpaca-18k",
        "stack-smol", "python-25k", "alpaca-120k", "code-120k"
    ]
    for (part_name, count), sname in zip(labels, source_names):
        print(f"  {sname:<20} : {count:>8,}")
    print(f"  {'─'*32}")
    print(f"  {'TOTAL':<20} : {len(combined):>8,}")

    combined.save_to_disk(str(SAVE_PATH))

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"Complete in {elapsed:.1f} minutes.")
    print(f"  Rows     : {len(combined):,}")
    print(f"  Location : {SAVE_PATH}")

    print("\nSpot check (3 samples):")
    for i in [0, len(combined)//2, len(combined)-1]:
        c = combined[i]["code"]
        print(f"  [{i}] {len(c)} chars | {repr(c[:65])}...")

    shutil.rmtree(PARTS_PATH, ignore_errors=True)

    print("\nNext steps:")
    print("  python -m data.preprocess")
    print("  python -m training.train")


if __name__ == "__main__":
    main()
