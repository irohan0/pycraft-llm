# training/lr_scheduler.py
#
# Cosine learning rate scheduler with linear warmup.
# Standard recipe for LLM pretraining in 2025.
#
# Schedule:
#   Steps 0 → warmup_steps : linear ramp  0 → max_lr
#   Steps warmup → max_steps: cosine decay max_lr → min_lr

import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,   # min_lr = max_lr * min_lr_ratio
) -> LambdaLR:
    """
    Returns a LambdaLR scheduler implementing:
      - Linear warmup from 0 to max_lr over warmup_steps
      - Cosine decay from max_lr to min_lr over remaining steps

    Args:
        optimizer     : the AdamW optimizer
        warmup_steps  : number of linear warmup steps
        max_steps     : total training steps
        min_lr_ratio  : floor as fraction of peak lr (default 0.1 = 10%)
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        # Cosine annealing: goes from 1.0 → min_lr_ratio
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale so it never goes below min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
