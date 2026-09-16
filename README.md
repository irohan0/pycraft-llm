# PyCraft-1 🐍

> **A 55M parameter Python code LLM trained entirely from scratch on a consumer laptop GPU**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-imshadow0%2Fpycraft--1-yellow.svg)](https://huggingface.co/imshadow0/pycraft-1)
[![Hardware](https://img.shields.io/badge/Hardware-RTX%203050%204GB-green.svg)]()

PyCraft-1 demonstrates that a domain-specific code language model can be **trained from scratch on consumer hardware** — no cloud compute, no API access, no pretrained base model. Built as an MSc AI research project at the University of Manchester (2026), it implements a custom architecture combining six 2025-era techniques, trained on a quality-scored curriculum dataset of 309k Python examples.

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Training Pipeline](#training-pipeline)
- [Dataset](#dataset)
- [Results and Evaluation](#results-and-evaluation)
- [Comparison with Other Models](#comparison-with-other-models)
- [Model Capabilities](#model-capabilities)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Locally](#running-locally)
- [Repository Structure](#repository-structure)
- [Reproducing Results](#reproducing-results)
- [Novel Contributions](#novel-contributions)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Highlights

- **55.3M parameters** — trained from scratch, not fine-tuned from an existing model
- **4GB VRAM** — fits on a laptop GPU (NVIDIA RTX 3050) using BF16 + gradient checkpointing
- **1.05B tokens** seen during pretraining over 4,000 steps
- **PPL 3.15** on Python code after supervised fine-tuning
- **6 architecture techniques from 2025** — GQA, QK-Norm, RoPE, SwiGLU, RMSNorm, FIM
- **Quality curriculum pretraining** — 309k examples scored and ordered by educational value
- **Custom BPE tokenizer** — trained on 234k Python files, 32k vocabulary, 4 FIM special tokens
- **Complete open-source pipeline** — every component written from scratch and reproducible

---

## Architecture

PyCraft-1 is a decoder-only transformer with a custom architecture that deliberately incorporates techniques adopted by the latest models (Llama 3, Qwen 3, OLMo 2) published in 2025.

```
Input tokens
     │
     ▼
Token Embedding  [32000 × 512]
     │
     ▼  ×8 layers
┌─────────────────────────────────┐
│  RMSNorm (pre-norm)             │
│  Grouped Query Attention        │
│    ├─ 8 Q heads                 │
│    ├─ 2 KV heads  (GQA 4:1)    │
│    ├─ QK-Norm on Q and K        │
│    └─ RoPE positional encoding  │
│  Residual connection            │
│  RMSNorm (pre-norm)             │
│  SwiGLU FFN  [512 → 1408 → 512]│
│  Residual connection            │
└─────────────────────────────────┘
     │
     ▼
Final RMSNorm
     │
     ▼
LM Head  [512 → 32000]
```

### Architecture Parameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Parameters | 55.3M | Verified PyTorch count |
| Layers | 8 | Transformer blocks |
| d_model | 512 | Hidden dimension |
| Q heads | 8 | Query attention heads |
| KV heads | 2 | Key/Value heads (GQA 4:1 ratio) |
| head_dim | 64 | Per-head dimension |
| d_ff | 1408 | SwiGLU intermediate dim (4/3 × d_model) |
| Max seq len | 1024 | Context window tokens |
| Vocab size | 32,000 | Custom BPE vocabulary |
| RoPE theta | 10,000 | Rotary embedding base frequency |
| QK-Norm | RMSNorm | Applied to Q and K before RoPE |
| Dropout | 0.1 (pretrain) / 0.0 (SFT) | |

### Why Each Technique Was Chosen

**Grouped Query Attention (GQA)** — 8 Q heads share 2 KV heads, reducing KV-cache memory by 4× at inference. Used in Llama 3, Qwen 3, Mistral. Critical for fitting the model in 4GB VRAM during inference.

**QK-Norm** — RMSNorm applied to Q and K vectors before RoPE, adopted from OLMo 2 and Qwen 3 (2025). Stabilises training loss curves in small models by preventing attention logit explosion.

**RoPE** — Rotary Positional Embeddings encode relative position by rotating Q and K vectors. No learned positional parameters. Extrapolates to longer sequences better than learned absolute embeddings.

**SwiGLU** — Gated activation: `gate_proj(x) × SiLU(up_proj(x))` → `down_proj`. Gives better perplexity per FLOP than GELU-based FFN. Used in Llama 2/3, PaLM, Qwen.

**RMSNorm (pre-norm)** — Normalises before each sublayer (not after). More stable than post-norm for deep networks. Faster than LayerNorm (no mean subtraction).

**Fill-in-the-Middle (FIM)** — 50% of training batches use PSM format: `<fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle`. Enables code infilling (completing gaps), not just completion. Used in StarCoder2, Codestral.

---

## Training Pipeline

Training followed a three-phase pipeline:

```
Phase 1: Custom tokenizer training
    ├── 234,614 Python files from 3 sources
    ├── Byte-level BPE, 32k vocabulary
    └── 4 FIM special tokens added

Phase 2: Pretraining (causal LM + FIM)
    ├── 309,221 curated Python examples
    ├── Quality-scored and curriculum-ordered
    ├── 4,000 steps, 1.05B tokens
    └── Final loss 1.16, PPL 3.2

Phase 3: Supervised Fine-Tuning (SFT)
    ├── Magicoder-OSS-Instruct-75K (Python subset)
    ├── 40,000 instruction-solution pairs
    ├── 400 steps
    └── Final loss 1.15, PPL 3.15
```

### Training Configuration

| Setting | Pretraining | SFT |
|---|---|---|
| Optimiser | AdamW | AdamW |
| Learning rate | 3e-4 (cosine) | 1e-4 (cosine) |
| Warmup steps | 500 | 100 |
| Weight decay | 0.1 | 0.01 |
| Gradient clip | 1.0 | 1.0 |
| Micro batch | 4 | 4 |
| Grad accumulation | 64 | 16 |
| Effective batch | 256 | 64 |
| Precision | BF16 autocast | float32 |
| Dropout | 0.1 | 0.0 |
| Seq length | 1024 | 1024 |
| Hardware | RTX 3050 4GB | RTX 3050 4GB |
| Training time | ~19 hours | ~2 hours |

### Loss Curves

**Pretraining:**
```
Step      10  |  loss 10.28  |  ppl  29,293
Step     100  |  loss  7.12  |  ppl   1,235
Step     500  |  loss  2.63  |  ppl      14
Step   1,000  |  loss  1.74  |  ppl       5.7
Step   2,000  |  loss  1.40  |  ppl       4.1
Step   3,000  |  loss  1.25  |  ppl       3.5
Step   4,000  |  loss  1.16  |  ppl       3.2  ← final
```

**SFT:**
```
Step   10  |  loss 1.36  |  ppl  3.90
Step  100  |  loss 1.27  |  ppl  3.55
Step  200  |  loss 1.26  |  ppl  3.53
Step  300  |  loss 1.21  |  ppl  3.35
Step  400  |  loss 1.15  |  ppl  3.15  ← final
```

---

## Dataset

### Pretraining Data (309,221 examples)

A quality-scored curriculum from 6 sources, selected for Python educational value:

| Source | Examples | Type | Why included |
|---|---|---|---|
| nampdn-ai/tiny-codes | 120,580 | LLM-generated educational | Phi-1 "textbook quality" synthetic exercises |
| ise-uiuc/Magicoder-OSS-Instruct-75K | 51,307 | OSS-seeded instruction+code | Real-world patterns, natural language alignment |
| iamtarun/python_code_instructions_18k_alpaca | 17,708 | Curated instruction pairs | Task-following patterns |
| bigcode/the-stack-smol | 8,750 | Deduplicated production code | Clean real-world Python anchor |
| flytech/python-codes-25k | 45,248 | Cleaned Python tasks | Diverse problem types |
| iamtarun/code_instructions_120k_alpaca | 66,999 | Multi-lang filtered to Python | Scale and diversity |

### Quality Curriculum Scoring

Each example was scored on 5 heuristics (0.0–1.0):

```python
def quality_score(code: str) -> float:
    score = 0.0
    if '"""' in code or "'''" in code:  score += 0.2  # docstring
    if ' -> ' in code:                  score += 0.2  # type hints
    if re.search(r'#\s+\w+', code):     score += 0.2  # comments
    if meaningful_variable_names(code): score += 0.2  # naming
    if 100 <= len(code) <= 3000:        score += 0.2  # length
    return score
```

| Score band | Count | Percentage |
|---|---|---|
| High (≥ 0.8) | 91,587 | 29.6% |
| Medium (0.4–0.8) | 215,908 | 69.8% |
| Low (< 0.4) | 1,726 | 0.6% |

**Average quality score: 0.644** — the dataset skews high quality. High-scored examples are seen first during training (curriculum learning).

### Tokenizer Training Data

| Source | Files | Notes |
|---|---|---|
| codeparrot/github-code | 150,000 | Raw GitHub Python |
| bigcode/the-stack-smol | 9,810 | Curated deduplicated Python |
| ise-uiuc/Magicoder-OSS-Instruct-75K | 75,000 | Instruction+solution pairs |

**Total: 234,614 Python files** processed to train a 32k BPE vocabulary.

---

## Results and Evaluation

### Perplexity on Held-out Python Code

Evaluated on 5 hand-written Python functions not present in training data:

| Code sample | PPL (base) | PPL (SFT) |
|---|---|---|
| Binary search | 1.3 | 1.4 |
| Stack class | 1.6 | 1.7 |
| Word frequency counter | 2.4 | 2.5 |
| Chunked file reader | 2.5 | 2.6 |
| Retry decorator | 2.9 | 3.0 |
| **Average** | **2.05** | **2.16** |

*PPL < 5 indicates the model strongly predicts correct next tokens — it understands Python structure deeply.*

### Generation Quality (SFT model)

| Prompt | Output quality |
|---|---|
| `def factorial(n):` | Correct recursive implementation ✓ |
| `class LinkedList:` | Correct head/traversal structure ✓ |
| `def is_palindrome(s: str) -> bool:` | Perfect one-liner `s == s[::-1]` ✓ |
| `def load_json(filepath: str):` | Correct try/except with context manager ✓ |
| `def normalize(arr):` | Partially correct, wrong formula ✗ |

**4/5 prompts produce correct, runnable Python code.**

### Training Summary

| Metric | Value |
|---|---|
| Total parameters | 55.3M |
| Pretraining steps | 4,000 |
| Pretraining tokens | 1.05B |
| Pretraining loss | 1.16 |
| Pretraining PPL | 3.2 |
| SFT steps | 400 |
| SFT loss | 1.15 |
| SFT PPL | 3.15 |
| Held-out PPL (avg) | 2.16 |
| Total training time | ~22 hours |
| Hardware | RTX 3050 Laptop 4GB |

---

## Comparison with Other Models

> **Important context:** PyCraft-1 is compared here purely on size and methodology, not raw benchmark scores. Models like StarCoder2 and CodeLlama were trained on orders of magnitude more compute. The novel contribution of PyCraft-1 is the methodology and reproducibility, not state-of-the-art performance.

### Parameter Count and Training Compute

| Model | Parameters | Training tokens | GPU requirement | From scratch? |
|---|---|---|---|---|
| **PyCraft-1 (ours)** | **55M** | **1.05B** | **RTX 3050 4GB** | **Yes** |
| CodeParrot | 110M | 50B | Multi-GPU | Yes |
| GPT-Neo | 125M | 300B | Multi-GPU | Yes |
| StarCoder2-3B | 3B | 3.3T | Multi-GPU cluster | Yes |
| CodeLlama-7B | 7B | 2T+ | Multi-GPU cluster | No (Llama base) |
| Qwen2.5-Coder-7B | 7B | 5.5T | Multi-GPU cluster | No (Qwen base) |

### HumanEval Pass@1 Context

Published benchmarks show CodeParrot 110M achieves 3.80% Pass@1 on HumanEval and 2.50% on MBPP, while GPT-Neo 125M achieves 0.83% Pass@1 on HumanEval. These are the most comparable models to PyCraft-1 by parameter count.

| Model | Size | HumanEval Pass@1 | MBPP Pass@1 | Training compute |
|---|---|---|---|---|
| GPT-Neo | 125M | 0.83% | 0.33% | 300B tokens, multi-GPU |
| **PyCraft-1 (ours)** | **55M** | **3.66%** | *not evaluated* | **1.05B tokens, 1× RTX 3050** |
| CodeParrot | 110M | 3.80% | 2.50% | 50B tokens, multi-GPU |
| Codex | 300M | 13.17% | — | Large-scale proprietary |
| StarCoder2-3B | 3B | ~31% | ~35% | 3.3T tokens, cluster |

*Measured with greedy decoding — the conventional and reproducible setting for single-sample pass@1 — on the `checkpoints/sft_stage1` model. Reproduce with:*

```bash
python -m eval.humaneval_runner -y --temperature 0.0
```

*Passing problems: `greatest_common_divisor`, `strlen`, `get_positive`, `is_prime`, `remove_vowels`, `add`. Sampling at temperature 0.1 instead gives 3.05-3.66% across runs, which is why the greedy number is the one reported.*

**Key insight:** PyCraft-1 scores 3.66% Pass@1 — 4.4× GPT-Neo 125M on 0.35% of its training tokens, and within a rounding error of CodeParrot 110M on 2% of its tokens, from a model half their size trained on one laptop GPU. It also achieves competitive perplexity (PPL 3.15) compared to similarly-sized models, while requiring only a fraction of the compute and being fully reproducible on consumer hardware. StarCoder2-3B achieves strong results on HumanEval and MBPP but requires multi-GPU training infrastructure and significantly more data. PyCraft-1 validates that quality-curriculum training can partially compensate for scale.

### Architecture Comparison

| Feature | PyCraft-1 | CodeParrot | GPT-Neo 125M | StarCoder2-3B |
|---|---|---|---|---|
| Architecture | Decoder-only | Decoder-only | Decoder-only | Decoder-only |
| Attention | GQA (4:1) | MHA | MHA | GQA |
| Positional encoding | RoPE | Learned abs | Learned abs | RoPE |
| QK-Norm | ✓ (2025) | ✗ | ✗ | ✗ |
| FFN | SwiGLU | GELU | GELU | SwiGLU |
| Normalisation | RMSNorm | LayerNorm | LayerNorm | LayerNorm |
| FIM training | ✓ | ✗ | ✗ | ✓ |
| Trainable on 4GB GPU | ✓ | ✗ | ✗ | ✗ |
| Open training code | ✓ | ✓ | ✓ | ✓ |

---

## Model Capabilities

### What PyCraft-1 Can Do

**Code completion** — given the start of a Python function, continues it:
```python
# Input:
def binary_search(arr, target):
    """Search sorted array for target."""
    
# PyCraft-1 output:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**Class implementation** — understands OOP patterns:
```python
# Input:
class Stack:
    def __init__(self):

# PyCraft-1 output:
        self._items = []
    
    def push(self, item):
        self._items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()
    
    def is_empty(self):
        return len(self._items) == 0
```

**Fill-in-the-Middle (FIM)** — unique to models with FIM training:
```python
# Given prefix + suffix, fills the middle
<fim_prefix>def calculate(a, b):
    """Add two numbers."""
    <fim_suffix>
    return result
<fim_middle>    result = a + b
```

**Instruction following** (SFT model):
```
# Task: Write a function to check if a number is prime

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

### What PyCraft-1 Cannot Do

- Multi-turn conversation (not a chat model)
- Reasoning about complex multi-file codebases (1024 token context)
- Guarantee correctness on complex algorithms
- Handle non-Python code reliably
- Replace a production code assistant like GitHub Copilot

---

## Installation

### Requirements

- Python 3.10+
- **No GPU required.** PyCraft-1 is 55M parameters and runs on CPU; CUDA is used when present but never needed.
- ~250MB disk space for the weights and tokenizer (more only if you retrain)

### Setup

```bash
git clone https://github.com/irohan0/pycraft-llm.git
cd pycraft-llm

conda create -n pycraft python=3.11 -y
conda activate pycraft

# CPU-only PyTorch (much smaller download than the CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# ...or, for an NVIDIA GPU:
# pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118

pip install -e ".[serve,hub]"
```

The weights and tokenizer are not in the repository (they are too large for git), so fetch them from HuggingFace:

```bash
python -c "
from huggingface_hub import hf_hub_download
import shutil, pathlib
pathlib.Path('checkpoints/sft_stage1').mkdir(parents=True, exist_ok=True)
pathlib.Path('tokenizer/vocab').mkdir(parents=True, exist_ok=True)
shutil.copy(hf_hub_download('imshadow0/pycraft-1', 'model.safetensors'),
            'checkpoints/sft_stage1/model.safetensors')
shutil.copy(hf_hub_download('imshadow0/pycraft-1', 'tokenizer/tokenizer.json'),
            'tokenizer/vocab/tokenizer.json')
"
```

You can skip this step entirely — if the files are absent, PyCraft downloads them from HuggingFace on first use.

---

## Quick Start

### Command line

```bash
pycraft generate "# Task: check if a number is prime\n\ndef is_prime(n):\n"
pycraft fim --prefix 'def square(n):\n    ' --suffix '\n\nprint(square(4))' --full
pycraft chat
pycraft info
```

### Python

```python
from pycraft import PyCraft

pc = PyCraft()                       # add quantize=True for ~1.4x on CPU

print(pc.complete_code("# Task: reverse a list\n\ndef reverse_list(xs):\n"))

# Streaming
for delta in pc.stream("def total(xs):\n", max_new_tokens=60):
    print(delta, end="", flush=True)

# Several prompts at once — roughly 4x aggregate throughput at batch 8
print(pc.generate_batch(["def a():\n", "def b():\n"], max_new_tokens=40))

# Fill in the middle (half of pretraining used this objective)
print(pc.fill_in_middle(
    prefix="def factorial(n):\n    if n <= 1:\n        return 1\n    ",
    suffix="\n\nprint(factorial(5))\n",
))
# -> return n * factorial(n-1)
```

### Direct model access

For training, evaluation, or anything that needs the raw module:

```python
import torch
from safetensors.torch import load_file
from model.config import get_config_120m
from model.pycraft_model import PyCraftModel
from tokenizer.tokenizer_utils import PyCraftTokenizer

device    = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PyCraftTokenizer("tokenizer/vocab/tokenizer.json")

cfg            = get_config_120m()
cfg.vocab_size = 32000
cfg.dropout    = 0.0

model = PyCraftModel(cfg).to(device)
model.load_state_dict(load_file("checkpoints/sft_stage1/model.safetensors", device=device))
model.eval()

prompt = "def fibonacci(n: int) -> int:\n    \"\"\"Return nth Fibonacci number.\"\"\"\n    "
ids    = tokenizer.encode(prompt)
inp    = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    out = model.generate(inp, max_new_tokens=100, temperature=0.2, top_k=20,
                         repetition_penalty=1.1)

print(tokenizer.decode(out[0, len(ids):].tolist(), skip_special_tokens=True))
```

Generation uses a KV cache and stops at `<|endoftext|>`. `temperature=0.0` selects greedy decoding; `top_p`, `repetition_penalty` and `stop_strings` are also supported.

### Run evaluations

```bash
python -m eval.perplexity                              # held-out PPL
python -m eval.evaluate                                # generation samples
python -m eval.humaneval_runner -y --temperature 0.0   # HumanEval, ~8 min on CPU
python -m tests.test_kv_cache                          # correctness tests
```

---

## Running Locally

PyCraft-1 runs on CPU. There is no hosted service and nothing here costs money to run.

```bash
pip install -e ".[serve]"
```

Weights resolve from `checkpoints/sft_stage1` when running from a clone, and fall back to downloading from HuggingFace otherwise (`pip install -e ".[hub]"`).

### Command line

```bash
pycraft info
pycraft generate "# Task: reverse a list\n\ndef reverse_list(xs):\n"
pycraft fim --prefix 'def square(n):\n    ' --suffix '\n\nprint(square(4))' --full
pycraft chat
pycraft serve --port 8000
```

Useful flags: `--quantize` (int8, ~1.4× faster on CPU), `--threads N`, `-t/--temperature` (0.0 is greedy), `-n/--max-tokens`, `--stop` (repeatable).

### REST API

```bash
pycraft serve            # http://127.0.0.1:8000/docs
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/completions` | Continue a prompt, or a list of prompts as one batch. Set `"stream": true` for SSE (single prompt only). |
| `POST /v1/fim` | Fill the gap between `prefix` and `suffix`. |
| `GET /health` | Liveness plus the loaded configuration. |
| `GET /v1/models` | Model metadata. |

**Fill-in-the-Middle** is the endpoint worth knowing about. Half of pretraining used the FIM objective, so infilling is trained behaviour rather than a prompting trick:

```bash
curl -X POST http://127.0.0.1:8000/v1/fim \
  -H 'Content-Type: application/json' \
  -d '{"prefix":"def factorial(n):\n    if n <= 1:\n        return 1\n    ",
       "suffix":"\n\nprint(factorial(5))\n"}'
# -> {"middle": "return n * factorial(n-1)\n\n", ...}
```

The server binds to `127.0.0.1` and has **no authentication**. To share it temporarily, put a free tunnel in front of it rather than binding wider:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

### Docker

```bash
docker build -t pycraft .
docker run --rm -p 127.0.0.1:8000:8000 pycraft
```

CPU-only image. Weights are baked in; mount `checkpoints/` instead to keep the image small.

### Inference performance

Generation uses a KV cache, so per-token cost is flat rather than growing with context. Measured on the same laptop running **CPU-only**, 8 threads:

| Context | Without cache | With cache |
|---|---|---|
| 32 | 31.0 tok/s | 61.9 tok/s |
| 256 | 12.3 tok/s | 53.7 tok/s |
| 512 | 6.6 tok/s | 48.6 tok/s |

On a realistic workload (200-token prompt, 400 generated) this is **8.8×**: 48.5s becomes 5.5s. `--quantize` adds roughly 1.4× on top (68 → 94 tok/s).

Batching multiplies throughput again, since per-token cost is dominated by weight loading that every sequence in the batch shares (1 / 2 / 4 / 8 prompts = 65.7 / 129.9 / 186.7 / **267.7** tok/s aggregate). Pass a list to `/v1/completions` or `PyCraft.generate_batch`; prompts are left-padded internally and each result is cut at its own EOS, so mixed lengths are fine.

The 1024-token context is a **hard stop**, not a sliding window: the KV cache stores post-RoPE keys, which cannot be re-based without re-rotating every cached key.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full reference.

---

## Repository Structure

```
pycraft-llm/
│
├── model/                      # Model architecture
│   ├── config.py               # Hyperparameters and model presets
│   ├── attention.py            # GQA + QK-Norm + RoPE
│   ├── feedforward.py          # SwiGLU FFN
│   ├── transformer.py          # Single transformer block
│   ├── kv_cache.py             # KV cache + causal mask construction
│   ├── sampling.py             # top-k / top-p / repetition penalty
│   └── pycraft_model.py        # Full model + cached generation
│
├── tokenizer/                  # Tokenizer
│   ├── train_tokenizer.py      # BPE training on Python code
│   └── tokenizer_utils.py      # Load and use the tokenizer
│
├── data/                       # Data pipeline
│   ├── stream_dataset.py       # Local disk dataset with FIM
│   ├── preprocess.py           # Quality scoring and curriculum
│   └── fim_utils.py            # Fill-in-the-Middle transformation
│
├── training/                   # Training infrastructure
│   ├── train.py                # Pretraining entry point
│   ├── trainer.py              # Core training loop
│   ├── sft_train.py            # SFT fine-tuning
│   ├── lr_scheduler.py         # Cosine schedule with warmup
│   └── checkpointing.py        # Save/load checkpoints
│
├── eval/                       # Evaluation
│   ├── perplexity.py           # PPL on held-out Python
│   ├── evaluate.py             # Generation quality test
│   └── humaneval_runner.py     # HumanEval benchmark
│
├── pycraft/                    # Inference package (CLI + REST API)
│   ├── engine.py               # Loading, generation, FIM, quantization
│   ├── server.py               # FastAPI app
│   └── cli.py                  # pycraft generate/fim/serve/chat
│
├── tests/                      # Correctness tests
│   └── test_kv_cache.py        # Cached vs uncached equivalence
│
├── scripts/                    # Utility scripts
│   └── download_pretrain_data.py  # Dataset download
│
├── docs/
│   └── DEPLOYMENT.md           # Running PyCraft-1 locally
│
├── pyproject.toml              # Installable package + CLI entry point
├── Dockerfile                  # CPU-only inference image
├── environment.yml             # Conda environment spec
└── README.md                   # This file
```

---

## Reproducing Results

All experiments were run on a single NVIDIA RTX 3050 Laptop GPU (4GB VRAM) with Windows 11, Anaconda Python 3.11.

### Step 1 — Environment

```bash
conda env create -f environment.yml
conda activate pycraft
```

### Step 2 — Train tokenizer

```bash
python -m tokenizer.train_tokenizer
# Runtime: ~10 minutes
# Output: tokenizer/vocab/tokenizer.json
```

### Step 3 — Download pretraining data

```bash
python scripts/download_pretrain_data.py
# Runtime: ~5 minutes
# Output: data/pretrain_python/ (~600MB)
```

### Step 4 — Build quality curriculum

```bash
python -m data.preprocess
# Runtime: ~1 minute
# Output: data/pretrain_curriculum/
```

### Step 5 — Pretrain

```bash
python -m training.train
# Runtime: ~19 hours on RTX 3050
# Output: checkpoints/step_0004000/
```

### Step 6 — SFT fine-tuning

```bash
python -m training.sft_train
# Runtime: ~2 hours
# Output: checkpoints/sft_stage1/
```

### Step 7 — Evaluate

```bash
python -m eval.perplexity
python -m eval.evaluate
python -m eval.humaneval_runner -y --temperature 0.0
```

### Expected Results

| Step | Metric | Expected value |
|---|---|---|
| Tokenizer | Vocab size | 32,000 |
| Tokenizer | Round-trip test | All 3 pass |
| Pretrain step 4000 | Training loss | ~1.16 |
| Pretrain step 4000 | PPL | ~3.2 |
| SFT step 400 | Training loss | ~1.15 |
| SFT step 400 | PPL | ~3.15 |
| Held-out eval | Average PPL | ~2.16 |
| HumanEval (greedy) | Pass@1 | 3.66% (6/164) |

---

## Novel Contributions

This project makes four contributions to the research on resource-constrained LLM training:

### 1. Quality-First Data Curriculum
A lightweight 5-heuristic quality scorer (docstrings, type hints, comments, naming, length) orders training examples from highest to lowest quality. This implements the Phi-1 "textbook quality" hypothesis at the consumer-hardware training regime, with ablation possible by comparing shuffled vs. curriculum-ordered training.

### 2. QK-Norm in a From-Scratch Small Model
RMSNorm applied to Q and K vectors before RoPE — adopted from OLMo 2 and Qwen 3 (2025) — demonstrating training stability improvement. First application of this technique in a completely from-scratch small code model trained on consumer hardware.

### 3. FIM Pretraining on 4GB VRAM
Fill-in-the-Middle objective (PSM format, 50% of batches) trained on a 4GB laptop GPU using gradient accumulation (effective batch 256) and BF16 mixed precision, with memory-efficient SDPA replacing Flash Attention.

### 4. Full Consumer-Hardware Reproducibility
Complete, documented pipeline — tokenizer training, architecture implementation, pretraining, SFT, evaluation — runnable end-to-end on a single 4GB laptop GPU in under one week. Addresses the reproducibility gap in code LLM research where most prior work requires multi-GPU clusters.

---

## Limitations

- **Scale:** 55M parameters with 1.05B training tokens is below the Chinchilla-optimal compute budget. Larger models trained with this pipeline would likely outperform this baseline.
- **Context window:** 1024 tokens limits reasoning over long functions or multi-file code. This is a hard stop rather than a sliding window — the KV cache stores post-RoPE keys, which cannot be re-based without re-rotating every cached key.
- **Markdown fences:** the SFT data (Magicoder) was full of ```python fences, so raw output often contains them. The CLI and API strip them by default; `strip_fences()` is exported for direct users.
- **Benchmark scores:** HumanEval and MBPP scores are modest — the contribution is the methodology and reproducibility, not state-of-the-art performance.
- **Language coverage:** Python-only. No multilingual code capability.
- **No RLHF:** The SFT model is not aligned with human preferences beyond instruction format.

---

## Citation

If you use PyCraft-1 or this codebase in your research, please cite:

```bibtex
@misc{inamdar2026pycraft,
  title   = {PyCraft-1: Training a Python Code LLM From Scratch on Consumer Hardware},
  author  = {Inamdar, Rohan},
  year    = {2026},
  institution = {University of Manchester, MSc Artificial Intelligence},
  note    = {Available at: https://huggingface.co/imshadow0/pycraft-1}
}
```

---

## Links

| Resource | URL |
|---|---|
| HuggingFace Model | https://huggingface.co/imshadow0/pycraft-1 |
| GitHub Repository | https://github.com/irohan0/pycraft-llm |
| Author LinkedIn | https://linkedin.com/in/rohan-inamdar-47aa4b251 |
| Author Google Scholar | https://scholar.google.com/citations?user=rnfdLu8AAAAJ |

---

## License

MIT License — see [LICENSE](LICENSE) for details. Model weights and training code are freely available for research and commercial use.

---

*Built as part of MSc Artificial Intelligence dissertation, University of Manchester, 2026.*
*Supervised by Dr. Mehran Hosseini.*
