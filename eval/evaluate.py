# eval/evaluate.py
#
# Generation quality evaluation for PyCraft-1.
# Tests the model's ability to complete Python code prompts.
# Run after any checkpoint to see qualitative output.
#
# python -m eval.evaluate

import torch
from pathlib import Path
from safetensors.torch import load_file

from model.pycraft_model import PyCraftModel
from model.config import get_config_120m, get_config_tiny
from tokenizer.tokenizer_utils import PyCraftTokenizer


# Prompts to complete — diverse Python patterns
PROMPTS = [
    "def factorial(n):\n    \"\"\"Return the factorial of n.\"\"\"\n",
    "class LinkedList:\n    def __init__(self):\n",
    "import numpy as np\n\ndef normalize(arr):\n    \"\"\"Normalize array to [0, 1] range.\"\"\"\n",
    "def is_palindrome(s: str) -> bool:\n",
    "# Read a JSON file and return its contents\ndef load_json(filepath: str):\n",
]


@torch.no_grad()
def generate_completions(
    model: PyCraftModel,
    tokenizer: PyCraftTokenizer,
    prompts: list[str],
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = "cuda",
) -> list[str]:
    """Generate a completion for each prompt."""
    model.eval()
    completions = []

    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # Decode only the newly generated tokens
        new_ids = output_ids[0, len(ids):].tolist()
        completion = tokenizer.decode(new_ids)
        completions.append(completion)

    return completions


def run_evaluation(
    checkpoint_path: str | Path | None = None,
    use_tiny: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("PyCraft-1 Generation Quality Evaluation")
    print("=" * 60)

    tokenizer = PyCraftTokenizer()

    cfg = get_config_tiny() if use_tiny else get_config_120m()
    cfg.vocab_size = tokenizer.vocab_size
    model = PyCraftModel(cfg).to(device)

    if checkpoint_path is not None:
        weights_path = Path(checkpoint_path) / "model.safetensors"
        weights = load_file(str(weights_path), device=device)
        model.load_state_dict(weights)
        print(f"Loaded: {Path(checkpoint_path).name}\n")
    else:
        print("No checkpoint — using random weights (output will be noise).\n")

    completions = generate_completions(
        model, tokenizer, PROMPTS, device=device,
        max_new_tokens=80, temperature=0.7, top_k=40,
    )

    for i, (prompt, completion) in enumerate(zip(PROMPTS, completions), 1):
        print(f"--- Prompt {i} ---")
        print(prompt.rstrip())
        print(f"--- Completion ---")
        print(completion.rstrip())
        print()

    print("=" * 60)


if __name__ == "__main__":
    from pathlib import Path
    ckpt = Path("checkpoints/sft_stage1")
    if ckpt.exists():
        run_evaluation(checkpoint_path=ckpt, use_tiny=False)
    else:
        ckpt = Path("checkpoints/step_0004000")
        run_evaluation(checkpoint_path=ckpt, use_tiny=False)
