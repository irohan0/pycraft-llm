# eval/perplexity.py
#
# Evaluates PyCraft-1 perplexity on a held-out Python code sample.
# Perplexity = exp(average cross-entropy loss).
# Lower is better. A random model over 32k vocab has PPL ≈ 32,000.
# Well-trained small code models typically achieve PPL < 10 on Python.
#
# Run at any point during or after training:
#   python -m eval.perplexity

import math
import torch
from pathlib import Path
from safetensors.torch import load_file

from model.pycraft_model import PyCraftModel
from model.config import get_config_120m, get_config_tiny
from tokenizer.tokenizer_utils import PyCraftTokenizer


# A diverse set of real Python snippets for evaluation
# These are NOT in the training data — used only for evaluation
EVAL_SAMPLES = [
    # Standard algorithms
    """def binary_search(arr, target):
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
""",
    # Data processing pattern
    """import csv
from collections import defaultdict

def count_word_frequencies(filepath):
    frequencies = defaultdict(int)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            for word in row:
                frequencies[word.strip().lower()] += 1
    return dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
""",
    # Class definition
    """class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0

    def __len__(self):
        return len(self._items)
""",
    # Decorator pattern
    """import time
import functools

def retry(max_attempts=3, delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator
""",
    # Generator / iterator pattern
    """def read_in_chunks(file_path, chunk_size=1024):
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def process_large_file(file_path):
    total_bytes = 0
    for chunk in read_in_chunks(file_path):
        total_bytes += len(chunk)
    return total_bytes
""",
]


@torch.no_grad()
def compute_perplexity(
    model: PyCraftModel,
    tokenizer: PyCraftTokenizer,
    samples: list[str],
    device: str = "cuda",
    max_seq_len: int = 512,
) -> dict:
    """
    Compute per-sample and average perplexity.

    Returns dict with:
        per_sample_ppl  : list of PPL values, one per sample
        avg_ppl         : average PPL across all samples
        avg_loss        : average cross-entropy loss
        total_tokens    : total tokens evaluated
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    per_sample = []

    for sample in samples:
        ids = tokenizer.encode(sample)

        # Truncate to max_seq_len + 1 (need +1 for target shift)
        ids = ids[:max_seq_len + 1]
        if len(ids) < 2:
            continue

        input_ids = torch.tensor(
            ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
        labels = torch.tensor(
            ids[1:],  dtype=torch.long).unsqueeze(0).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(input_ids, labels)

        n_tokens = input_ids.shape[1]
        sample_ppl = math.exp(min(loss.item(), 20))
        per_sample.append(sample_ppl)
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    avg_ppl = math.exp(min(avg_loss, 20))

    return {
        "per_sample_ppl": per_sample,
        "avg_ppl":        avg_ppl,
        "avg_loss":       avg_loss,
        "total_tokens":   total_tokens,
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path | None = None,
    use_tiny: bool = True,
):
    """
    Load a checkpoint and evaluate perplexity.
    If checkpoint_path is None, evaluates the randomly-initialised model
    (useful as a baseline — should give PPL ≈ 32,000).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 55)
    print("PyCraft-1 Perplexity Evaluation")
    print("=" * 55)

    # Load tokenizer
    tokenizer = PyCraftTokenizer()
    print(f"Tokenizer: {tokenizer.vocab_size:,} vocab")

    # Build model
    cfg = get_config_tiny() if use_tiny else get_config_120m()
    cfg.vocab_size = tokenizer.vocab_size
    model = PyCraftModel(cfg).to(device)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        # Find model.safetensors inside the checkpoint directory
        weights_path = checkpoint_path / "model.safetensors"
        if not weights_path.exists():
            # Maybe checkpoint_path IS the weights file
            weights_path = checkpoint_path
        print(f"Loading weights from: {weights_path}")
        weights = load_file(str(weights_path), device=device)
        model.load_state_dict(weights)
        print("Weights loaded.")
    else:
        print("No checkpoint — evaluating random init (baseline).")

    # Evaluate
    print(f"\nEvaluating {len(EVAL_SAMPLES)} Python code samples...\n")
    results = compute_perplexity(model, tokenizer, EVAL_SAMPLES, device)

    print("Per-sample perplexity:")
    sample_names = [
        "binary_search",
        "word_frequencies",
        "Stack class",
        "retry decorator",
        "chunked file reader",
    ]
    for name, ppl in zip(sample_names, results["per_sample_ppl"]):
        bar = "█" * min(int(ppl / 50), 40)
        print(f"  {name:<22} PPL: {ppl:>8.1f}  {bar}")

    print(f"\nAverage PPL  : {results['avg_ppl']:.2f}")
    print(f"Average loss : {results['avg_loss']:.4f}")
    print(f"Total tokens : {results['total_tokens']:,}")
    print()

    # Interpretation guide
    ppl = results["avg_ppl"]
    if ppl > 10000:
        verdict = "Random init — model has not learned yet."
    elif ppl > 1000:
        verdict = "Early training — model learning basic structure."
    elif ppl > 100:
        verdict = "Mid training — model recognises Python patterns."
    elif ppl > 20:
        verdict = "Good — model produces reasonable Python code."
    elif ppl > 5:
        verdict = "Strong — model understands Python well."
    else:
        verdict = "Excellent — potential overfitting, check diversity."

    print(f"Verdict: {verdict}")
    print("=" * 55)
    return results


if __name__ == "__main__":
    from pathlib import Path
    ckpt = Path("checkpoints/sft_stage1")
    if ckpt.exists():
        evaluate_checkpoint(checkpoint_path=ckpt, use_tiny=False)
    else:
        ckpt = Path("checkpoints/step_0004000")
        evaluate_checkpoint(checkpoint_path=ckpt, use_tiny=False)
