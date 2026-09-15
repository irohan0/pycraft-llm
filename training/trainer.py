# training/trainer.py
#
# Core training loop for PyCraft-1.
# Fixed: duplicate checkpoint bug, log file closed before final save.

import time
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.pycraft_model import PyCraftModel
from model.config import PyCraftConfig
from training.checkpointing import (
    save_checkpoint, load_checkpoint,
    find_latest_checkpoint, TrainingState,
)
from training.lr_scheduler import get_cosine_schedule_with_warmup


@dataclass
class TrainerConfig:
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir:        str = "logs"

    # Optimiser
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.1
    beta1:          float = 0.9
    beta2:          float = 0.95
    grad_clip:      float = 1.0

    # Batch / accumulation
    micro_batch_size:            int = 4
    gradient_accumulation_steps: int = 64

    # Schedule
    warmup_steps: int = 2000
    max_steps:    int = 100_000

    # Logging / saving
    log_every:        int = 50
    checkpoint_every: int = 1000
    keep_checkpoints: int = 3

    # Resume
    resume: bool = True

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


class PyCraftTrainer:
    def __init__(
        self,
        model: PyCraftModel,
        train_loader: DataLoader,
        config: TrainerConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Separate weight decay: apply only to 2D+ params (not biases/norms)
        decay_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and p.dim() < 2]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            # fused AdamW is CUDA-only in torch 2.3; on CPU it raises at
            # construction, so gate it on the actual device.
            fused=(device == "cuda"),
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )

        self.state = TrainingState(
            step=0, epoch=0, best_loss=float("inf"), tokens_seen=0
        )

        # Open log file — stays open for the entire training run
        log_path = Path(config.log_dir) / "train.log"
        self.log_file = open(log_path, "a", buffering=1)  # line-buffered

    def _log(self, msg: str):
        """Print to console and write to log file."""
        print(msg)
        if not self.log_file.closed:
            self.log_file.write(msg + "\n")

    def _close_log(self):
        """Safely close the log file."""
        if not self.log_file.closed:
            self.log_file.flush()
            self.log_file.close()

    def _try_resume(self):
        """Auto-resume from latest checkpoint if one exists."""
        if not self.config.resume:
            self._log("Resume disabled — starting from scratch.")
            return

        ckpt = find_latest_checkpoint(self.config.checkpoint_dir)
        if ckpt is None:
            self._log("No checkpoint found — starting from scratch.")
            return

        self._log(f"Resuming from checkpoint: {ckpt.name}")
        self.state = load_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.config.checkpoint_dir, device=self.device,
        )
        self._log(
            f"Resumed at step {self.state.step:,}, "
            f"tokens seen: {self.state.tokens_seen/1e6:.1f}M"
        )

    def _save(self, step: int, tokens_seen: int):
        """Save a checkpoint and update training state."""
        self.state.step = step
        self.state.tokens_seen = tokens_seen
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.state, self.config.checkpoint_dir,
            keep_last_n=self.config.keep_checkpoints,
        )

    def train(self):
        self._try_resume()

        cfg = self.config
        model = self.model
        device = self.device

        model.train()
        data_iter = iter(self.train_loader)
        step = self.state.step
        tokens_seen = self.state.tokens_seen
        last_saved = step   # track which step was last checkpointed

        self._log("\n" + "=" * 60)
        self._log("PyCraft-1 Training Started")
        self._log(f"  Device           : {device}")
        self._log(
            f"  Parameters       : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        self._log(f"  Micro batch size : {cfg.micro_batch_size}")
        self._log(f"  Grad accumulation: {cfg.gradient_accumulation_steps}")
        self._log(f"  Effective batch  : {cfg.effective_batch_size}")
        self._log(f"  Max steps        : {cfg.max_steps:,}")
        self._log(f"  Warmup steps     : {cfg.warmup_steps:,}")
        self._log(f"  Starting at step : {step:,}")
        self._log("=" * 60 + "\n")

        t0 = time.time()

        try:
            while step < cfg.max_steps:
                self.optimizer.zero_grad(set_to_none=True)
                loss_accum = 0.0
                tokens_in_batch = 0

                # ---------------------------------------------------- #
                # Gradient accumulation inner loop
                # ---------------------------------------------------- #
                for _ in range(cfg.gradient_accumulation_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.train_loader)
                        batch = next(data_iter)

                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, loss = model(input_ids, labels)

                    # Average loss over accumulation steps
                    loss = loss / cfg.gradient_accumulation_steps
                    loss.backward()

                    loss_accum += loss.item()
                    tokens_in_batch += input_ids.numel()

                # ---------------------------------------------------- #
                # Optimiser step
                # ---------------------------------------------------- #
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
                self.optimizer.step()
                self.scheduler.step()

                step += 1
                tokens_seen += tokens_in_batch

                # ---------------------------------------------------- #
                # Logging
                # ---------------------------------------------------- #
                if step % cfg.log_every == 0:
                    t1 = time.time()
                    dt = max(t1 - t0, 1e-6)
                    t0 = t1
                    lr = self.optimizer.param_groups[0]["lr"]
                    ppl = math.exp(min(loss_accum, 20))
                    tok_per_s = (cfg.log_every
                                 * cfg.effective_batch_size
                                 * cfg.micro_batch_size) / dt

                    if loss_accum < self.state.best_loss:
                        self.state.best_loss = loss_accum

                    self._log(
                        f"step {step:>7,} | "
                        f"loss {loss_accum:.4f} | "
                        f"ppl {ppl:>9.1f} | "
                        f"lr {lr:.2e} | "
                        f"grad {grad_norm:.3f} | "
                        f"tok/s {tok_per_s:>8,.0f} | "
                        f"seen {tokens_seen/1e6:.1f}M"
                    )

                # ---------------------------------------------------- #
                # Periodic checkpoint
                # ---------------------------------------------------- #
                if step % cfg.checkpoint_every == 0:
                    self._save(step, tokens_seen)
                    last_saved = step

        except KeyboardInterrupt:
            self._log(f"\nInterrupted at step {step:,} — saving checkpoint...")

        finally:
            # Save final checkpoint only if we haven't just saved
            if step != last_saved:
                self._log("Saving final checkpoint...")
                self._save(step, tokens_seen)

            self._log(f"\nTraining complete.")
            self._log(f"  Final step    : {step:,}")
            self._log(f"  Tokens seen   : {tokens_seen/1e9:.4f}B")
            self._log(f"  Best loss     : {self.state.best_loss:.4f}")
            self._close_log()
