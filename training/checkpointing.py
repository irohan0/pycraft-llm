# training/checkpointing.py
#
# Handles saving and loading training checkpoints.
# Saves model weights, optimizer state, scheduler state,
# and training metadata so training can resume after interruption.

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from safetensors.torch import save_file, load_file


@dataclass
class TrainingState:
    """Everything needed to resume training from a checkpoint."""
    step: int
    epoch: int
    best_loss: float
    tokens_seen: int


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    state: TrainingState,
    checkpoint_dir: str | Path,
    keep_last_n: int = 3,
):
    """
    Save model + optimizer + scheduler + metadata.
    Uses safetensors for model weights (safer than .pt files).
    Keeps only the last N checkpoints to save disk space.
    """
    checkpoint_dir = Path(checkpoint_dir)
    step_dir = checkpoint_dir / f"step_{state.step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model weights → safetensors format
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        step_dir / "model.safetensors",
    )

    # 2. Optimizer + scheduler → standard torch format
    torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), step_dir / "scheduler.pt")

    # 3. Training metadata → JSON (human readable)
    with open(step_dir / "state.json", "w") as f:
        json.dump(asdict(state), f, indent=2)

    print(f"  Checkpoint saved → {step_dir}")

    # 4. Cleanup: remove oldest checkpoints beyond keep_last_n
    all_ckpts = sorted(checkpoint_dir.glob("step_*"), key=lambda p: p.name)
    for old in all_ckpts[:-keep_last_n]:
        import shutil
        shutil.rmtree(old)
        print(f"  Removed old checkpoint: {old.name}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoint_dir: str | Path,
    step: Optional[int] = None,
    device: str = "cuda",
) -> TrainingState:
    """
    Load the latest (or specified) checkpoint.
    Returns the TrainingState so training can resume correctly.
    """
    checkpoint_dir = Path(checkpoint_dir)

    if step is not None:
        step_dir = checkpoint_dir / f"step_{step:07d}"
    else:
        # Find the most recent checkpoint
        all_ckpts = sorted(checkpoint_dir.glob("step_*"), key=lambda p: p.name)
        if not all_ckpts:
            raise FileNotFoundError(
                f"No checkpoints found in {checkpoint_dir}")
        step_dir = all_ckpts[-1]

    print(f"  Loading checkpoint from {step_dir}")

    # Load model weights
    weights = load_file(step_dir / "model.safetensors", device=device)
    model.load_state_dict(weights)

    # Load optimizer and scheduler
    optimizer.load_state_dict(
        torch.load(step_dir / "optimizer.pt", map_location=device)
    )
    scheduler.load_state_dict(
        torch.load(step_dir / "scheduler.pt", map_location=device)
    )

    # Load training state
    with open(step_dir / "state.json") as f:
        state_dict = json.load(f)

    return TrainingState(**state_dict)


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """Returns path to latest checkpoint dir, or None if none exist."""
    checkpoint_dir = Path(checkpoint_dir)
    all_ckpts = sorted(checkpoint_dir.glob("step_*"), key=lambda p: p.name)
    return all_ckpts[-1] if all_ckpts else None
