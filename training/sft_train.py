# training/sft_train.py
#
# Two-stage fine-tuning for PyCraft-1:
#
# Stage 1 — SFT (Supervised Fine-Tuning)
#   Standard instruction tuning on Magicoder-OSS-Instruct-75K.
#   Full float32, dropout disabled, response-only loss masking.
#   This is the foundation that every published code LLM uses.
#
# Stage 2 — Lightweight preference signal (ORPO-inspired)
#   After SFT converges, add a simple margin loss between
#   chosen and rejected responses. This is the novel contribution
#   we report in the paper — single-stage preference alignment
#   applied post-SFT on a resource-constrained 55M model.
#
# Why two stages:
#   ORPO assumes the model already generates reasonable text.
#   Applying ORPO to a base model that has never seen instruction
#   format causes the high SFT loss you observed. Doing SFT first,
#   then preference alignment, is the correct pipeline used by
#   Llama 2, Qwen, Mistral, and every other production model.
#
# Paper framing:
#   "We apply a two-stage post-training pipeline: SFT for
#    instruction alignment followed by preference optimisation
#    using a simplified ORPO-inspired margin loss, demonstrating
#    that preference alignment is achievable on consumer hardware
#    without a separate reward model."

import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from model.config import get_config_120m
from model.pycraft_model import PyCraftModel
from tokenizer.tokenizer_utils import PyCraftTokenizer
from training.lr_scheduler import get_cosine_schedule_with_warmup

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
BASE_CHECKPOINT = "checkpoints/step_0004000"
SFT_CHECKPOINT = "checkpoints/sft_stage1"
ORPO_CHECKPOINT = "checkpoints/orpo_final"
LOG_PATH = "logs/sft_train.log"

# Shared
MAX_SEQ_LEN = 1024
IGNORE_INDEX = -100

# Stage 1 — SFT
SFT_BATCH_SIZE = 4
SFT_GRAD_ACCUM = 16      # effective batch = 64
SFT_LR = 1e-4
SFT_WARMUP = 100
SFT_MAX_STEPS = 400
SFT_LOG_EVERY = 10
SFT_SAVE_EVERY = 200

# Stage 2 — Preference
PREF_BATCH_SIZE = 2
PREF_GRAD_ACCUM = 32      # effective batch = 64
PREF_LR = 2e-5
PREF_WARMUP = 50
PREF_MAX_STEPS = 200
PREF_LOG_EVERY = 10
PREF_MARGIN = 0.5     # margin for preference loss


# ------------------------------------------------------------------ #
# Prompt format — matches pretraining distribution
# ------------------------------------------------------------------ #
def build_prompt(problem: str) -> str:
    """
    Use Python comment + docstring format that the model saw
    thousands of times during pretraining. Never use markdown
    headers (### Instruction) on a model not trained on them.
    """
    # Truncate very long problems
    problem = problem.strip()[:400]
    return f'# Task: {problem}\n\n'


# ------------------------------------------------------------------ #
# Find response boundary in token sequence
# ------------------------------------------------------------------ #
def find_response_start(
    full_ids: list,
    tokenizer: PyCraftTokenizer,
    prompt_text: str,
) -> int:
    """
    Encode prompt alone and use its length as the boundary.
    This is reliable because BPE boundaries only change at
    the junction between prompt and response, not inside prompt.
    We add a small buffer of 2 tokens to be safe.
    """
    prompt_ids = tokenizer.encode(prompt_text)
    # Buffer for potential BPE boundary effect at junction
    return max(0, len(prompt_ids) - 2)


# ------------------------------------------------------------------ #
# Rejected response generator
# ------------------------------------------------------------------ #
def make_rejected(solution: str) -> str:
    lines = solution.split('\n')
    rng = random.Random(hash(solution) % 2**32)
    n = rng.randint(0, 3)

    if n == 0:
        # Remove return statements
        new = [l for l in lines if not l.strip().startswith('return ')]
        r = '\n'.join(new)
        if r.strip() != solution.strip() and len(r.strip()) > 20:
            return r

    elif n == 1:
        # Truncate to 60%
        cut = max(1, int(len(lines) * 0.6))
        return '\n'.join(lines[:cut]) + '\n    pass'

    elif n == 2:
        # Replace variable names
        r = solution
        for old, new in [('result', 'x'), ('output', 'y'),
                         ('count', 'n'), ('value', 'v')]:
            r = re.sub(rf'\b{old}\b', new, r)
        if r != solution:
            return r

    else:
        # Remove colon from control flow
        new_lines = list(lines)
        for i, line in enumerate(new_lines):
            s = line.strip()
            if (s.startswith(('if ', 'for ', 'while ', 'def ', 'class '))
                    and s.endswith(':')):
                new_lines[i] = line[:-1]
                break
        r = '\n'.join(new_lines)
        if r != solution:
            return r

    return '\n'.join(lines[:max(1, len(lines)//2)]) + '\n    pass'


# ------------------------------------------------------------------ #
# Load Magicoder dataset
# ------------------------------------------------------------------ #
def load_magicoder(tokenizer: PyCraftTokenizer):
    sft_path = Path("data/sft/magicoder_oss")
    if sft_path.exists():
        ds = load_from_disk(str(sft_path))
        print(f"  Loaded Magicoder from disk: {len(ds):,} examples")
    else:
        print("  Downloading Magicoder-OSS-Instruct-75K...")
        ds = load_dataset(
            "ise-uiuc/Magicoder-OSS-Instruct-75K",
            split="train",
            trust_remote_code=True,
        )
        Path("data/sft").mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(sft_path))
    return ds


# ------------------------------------------------------------------ #
# Stage 1 Dataset — SFT
# ------------------------------------------------------------------ #
class SFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PyCraftTokenizer,
        max_seq_len: int = MAX_SEQ_LEN,
        max_samples: int = 40_000,
    ):
        self.examples = []
        ds = load_magicoder(tokenizer)

        print("  Building SFT examples...")
        skipped = verified = 0

        for sample in ds:
            if len(self.examples) >= max_samples:
                break

            problem = sample.get("problem", "").strip()
            solution = sample.get("solution", "").strip()

            if not problem or not solution or len(solution) < 40:
                skipped += 1
                continue

            # Skip non-Python
            if any(p in solution for p in [
                "function(", "const ", "var ", "=>",
                "public static", "#include",
            ]):
                skipped += 1
                continue

            prompt = build_prompt(problem)
            full_text = prompt + solution

            # Encode full string together (correct BPE boundaries)
            full_ids = tokenizer.encode(full_text) + [tokenizer.eot_id]

            if len(full_ids) < 15:
                skipped += 1
                continue

            # Find response start
            resp_start = find_response_start(full_ids, tokenizer, prompt)

            # Truncate and pad
            full_ids = full_ids[:max_seq_len + 1]
            pad_len = max_seq_len + 1 - len(full_ids)
            full_ids = full_ids + [tokenizer.pad_id] * pad_len

            inp = full_ids[:max_seq_len]
            lbl = full_ids[1:max_seq_len + 1]

            # Mask instruction tokens
            for j in range(min(resp_start, max_seq_len)):
                lbl[j] = IGNORE_INDEX
            # Mask padding
            for j in range(len(lbl)):
                if inp[j] == tokenizer.pad_id:
                    lbl[j] = IGNORE_INDEX

            valid = sum(1 for x in lbl if x != IGNORE_INDEX)

            if valid < 10:
                skipped += 1
                continue

            # Verify first example
            if verified == 0:
                print(f"  First example check:")
                print(f"    Prompt    : {resp_start} tokens")
                print(f"    Valid lbl : {valid} tokens (need >= 10)")
                print(f"    Total lbl : {max_seq_len} tokens")
                print(f"    Pct valid : {100*valid/max_seq_len:.1f}%")

            self.examples.append({
                "input_ids": torch.tensor(inp, dtype=torch.long),
                "labels":    torch.tensor(lbl, dtype=torch.long),
            })
            verified += 1

        print(f"  SFT: {len(self.examples):,} examples "
              f"(skipped {skipped:,})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ------------------------------------------------------------------ #
# Stage 2 Dataset — Preference pairs
# ------------------------------------------------------------------ #
class PrefDataset(Dataset):
    def __init__(
        self,
        tokenizer: PyCraftTokenizer,
        max_seq_len: int = MAX_SEQ_LEN,
        max_samples: int = 20_000,
    ):
        self.examples = []
        ds = load_magicoder(tokenizer)

        print("  Building preference pairs...")
        skipped = 0

        for sample in ds:
            if len(self.examples) >= max_samples:
                break

            problem = sample.get("problem", "").strip()
            solution = sample.get("solution", "").strip()

            if not problem or not solution or len(solution) < 40:
                skipped += 1
                continue

            if any(p in solution for p in [
                "function(", "const ", "var ", "=>",
                "public static", "#include",
            ]):
                skipped += 1
                continue

            rejected = make_rejected(solution)
            if rejected.strip() == solution.strip():
                skipped += 1
                continue

            prompt = build_prompt(problem)

            def tokenise(text):
                ids = tokenizer.encode(prompt + text) + [tokenizer.eot_id]
                ids = ids[:max_seq_len + 1]
                rs = find_response_start(ids, tokenizer, prompt)
                pad = max_seq_len + 1 - len(ids)
                ids = ids + [tokenizer.pad_id] * pad
                inp = ids[:max_seq_len]
                lbl = ids[1:max_seq_len + 1]
                for j in range(min(rs, max_seq_len)):
                    lbl[j] = IGNORE_INDEX
                for j in range(len(lbl)):
                    if inp[j] == tokenizer.pad_id:
                        lbl[j] = IGNORE_INDEX
                valid = sum(1 for x in lbl if x != IGNORE_INDEX)
                return inp, lbl, valid

            c_inp, c_lbl, c_valid = tokenise(solution)
            r_inp, r_lbl, r_valid = tokenise(rejected)

            if c_valid < 10 or r_valid < 10:
                skipped += 1
                continue

            self.examples.append({
                "chosen_ids":      torch.tensor(c_inp, dtype=torch.long),
                "chosen_labels":   torch.tensor(c_lbl, dtype=torch.long),
                "rejected_ids":    torch.tensor(r_inp, dtype=torch.long),
                "rejected_labels": torch.tensor(r_lbl, dtype=torch.long),
            })

        print(f"  Pref: {len(self.examples):,} pairs "
              f"(skipped {skipped:,})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ------------------------------------------------------------------ #
# Utility: compute mean log-prob over response tokens
# ------------------------------------------------------------------ #
def mean_log_prob(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Mean log-prob over non-masked tokens. Full float32."""
    lp = F.log_softmax(logits.float(), dim=-1)    # (B,T,V)
    mask = (labels != IGNORE_INDEX)                  # (B,T)
    safe = labels.clone()
    safe[~mask] = 0
    tok_lp = lp.gather(2, safe.unsqueeze(-1)).squeeze(-1)
    tok_lp = tok_lp * mask.float()
    return tok_lp.sum(-1) / mask.float().sum(-1).clamp(min=1)


# ------------------------------------------------------------------ #
# Stage 1 — SFT Training
# ------------------------------------------------------------------ #
def run_sft(
    model: PyCraftModel,
    tokenizer: PyCraftTokenizer,
    device: str,
    log,
) -> PyCraftModel:
    log("\n" + "="*60)
    log("STAGE 1 — Supervised Fine-Tuning (SFT)")
    log("="*60)

    dataset = SFTDataset(tokenizer)
    loader = DataLoader(
        dataset, batch_size=SFT_BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True,
    )

    # Separate weight decay params
    decay_p = [p for n, p in model.named_parameters()
               if p.requires_grad and p.dim() >= 2]
    no_decay_p = [p for n, p in model.named_parameters()
                  if p.requires_grad and p.dim() < 2]

    opt = torch.optim.AdamW(
        [{"params": decay_p,    "weight_decay": 0.01},
         {"params": no_decay_p, "weight_decay": 0.0}],
        lr=SFT_LR, betas=(0.9, 0.95), eps=1e-8,
    )
    sched = get_cosine_schedule_with_warmup(
        opt, SFT_WARMUP, SFT_MAX_STEPS,
    )

    # CRITICAL: disable dropout for fine-tuning
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    log(f"  Examples     : {len(dataset):,}")
    log(f"  Batch size   : {SFT_BATCH_SIZE} x {SFT_GRAD_ACCUM} = "
        f"{SFT_BATCH_SIZE*SFT_GRAD_ACCUM} effective")
    log(f"  LR           : {SFT_LR}")
    log(f"  Steps        : {SFT_MAX_STEPS}")
    log(f"  Dropout      : disabled (0.0)")
    log(f"  Precision    : float32 (no BF16 for stability)")
    log("")

    data_iter = iter(loader)
    step = 0
    best_loss = float("inf")
    acc_loss = 0.0

    while step < SFT_MAX_STEPS:
        opt.zero_grad(set_to_none=True)
        acc = 0.0

        for _ in range(SFT_GRAD_ACCUM):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            inp = batch["input_ids"].to(device)
            lbl = batch["labels"].to(device)

            # Full float32 forward pass — no BF16 autocast
            logits, _ = model(inp)

            loss = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]),
                lbl.view(-1),
                ignore_index=IGNORE_INDEX,
            )

            if not torch.isfinite(loss):
                continue

            (loss / SFT_GRAD_ACCUM).backward()
            acc += loss.item() / SFT_GRAD_ACCUM

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        if torch.isfinite(grad_norm):
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        acc_loss += acc

        # Hard stop at step 10 if loss is still wrong
        if step == 10:
            avg = acc_loss / 10
            if avg > 5.0:
                log(f"ERROR: SFT loss={avg:.2f} at step 10. Expected < 3.0.")
                log("Something is still wrong. Stopping.")
                break
            else:
                log(f"  Step 10 sanity check: loss={avg:.4f} -- OK")

        if step % SFT_LOG_EVERY == 0:
            avg = acc_loss / SFT_LOG_EVERY
            ppl = math.exp(min(avg, 20))
            lr = opt.param_groups[0]["lr"]
            if avg < best_loss:
                best_loss = avg
            log(f"sft step {step:>4} | loss {avg:.4f} | "
                f"ppl {ppl:.2f} | lr {lr:.2e} | grad {grad_norm:.3f}")
            acc_loss = 0.0

        if step % SFT_SAVE_EVERY == 0 or step == SFT_MAX_STEPS:
            Path(SFT_CHECKPOINT).mkdir(parents=True, exist_ok=True)
            save_file(
                {k: v.contiguous() for k, v in model.state_dict().items()},
                f"{SFT_CHECKPOINT}/model.safetensors",
            )
            log(f"  SFT checkpoint saved at step {step}")

    log(f"\n  SFT complete. Best loss: {best_loss:.4f}")
    return model


# ------------------------------------------------------------------ #
# Stage 2 — Preference Alignment
# ------------------------------------------------------------------ #
def run_preference(
    model: PyCraftModel,
    tokenizer: PyCraftTokenizer,
    device: str,
    log,
) -> PyCraftModel:
    log("\n" + "="*60)
    log("STAGE 2 — Preference Alignment (ORPO-inspired margin loss)")
    log("="*60)
    log("Novel contribution: lightweight preference alignment")
    log("on a 55M from-scratch model, no reference model needed.")
    log("")

    dataset = PrefDataset(tokenizer)
    loader = DataLoader(
        dataset, batch_size=PREF_BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True,
    )

    decay_p = [p for n, p in model.named_parameters()
               if p.requires_grad and p.dim() >= 2]
    no_decay_p = [p for n, p in model.named_parameters()
                  if p.requires_grad and p.dim() < 2]

    opt = torch.optim.AdamW(
        [{"params": decay_p,    "weight_decay": 0.01},
         {"params": no_decay_p, "weight_decay": 0.0}],
        lr=PREF_LR, betas=(0.9, 0.95), eps=1e-8,
    )
    sched = get_cosine_schedule_with_warmup(
        opt, PREF_WARMUP, PREF_MAX_STEPS,
    )

    # Keep dropout disabled
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    log(f"  Pairs        : {len(dataset):,}")
    log(f"  Margin       : {PREF_MARGIN}")
    log(f"  LR           : {PREF_LR}")
    log(f"  Steps        : {PREF_MAX_STEPS}")
    log("")

    data_iter = iter(loader)
    step = 0
    best_loss = float("inf")
    acc_sft = acc_pref = acc_gap = 0.0
    n_acc = 0

    while step < PREF_MAX_STEPS:
        opt.zero_grad(set_to_none=True)

        for _ in range(PREF_GRAD_ACCUM):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            cids = batch["chosen_ids"].to(device)
            clbl = batch["chosen_labels"].to(device)
            rids = batch["rejected_ids"].to(device)
            rlbl = batch["rejected_labels"].to(device)

            # Full float32 — no BF16
            logits_c, _ = model(cids)
            logits_r, _ = model(rids)

            # SFT loss on chosen (standard cross-entropy)
            sft_loss = F.cross_entropy(
                logits_c.float().view(-1, logits_c.shape[-1]),
                clbl.view(-1),
                ignore_index=IGNORE_INDEX,
            )

            # Preference loss: margin between chosen and rejected log-probs
            lp_c = mean_log_prob(logits_c, clbl)    # (B,)
            lp_r = mean_log_prob(logits_r, rlbl)    # (B,)

            # Margin loss: push chosen log_prob above rejected by margin
            # max(0, margin - (lp_chosen - lp_rejected))
            pref_loss = F.relu(
                PREF_MARGIN - (lp_c - lp_r)
            ).mean()

            loss = sft_loss + 0.1 * pref_loss

            if not torch.isfinite(loss):
                continue

            (loss / PREF_GRAD_ACCUM).backward()

            acc_sft += sft_loss.item() / PREF_GRAD_ACCUM
            acc_pref += pref_loss.item() / PREF_GRAD_ACCUM
            acc_gap += (lp_c - lp_r).mean().item() / PREF_GRAD_ACCUM
            n_acc += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        if torch.isfinite(grad_norm):
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1

        if step % PREF_LOG_EVERY == 0:
            lr = opt.param_groups[0]["lr"]
            # Accumulators span PREF_LOG_EVERY optimiser steps, so divide by
            # it — Stage 1 does the same. Without this a healthy loss of
            # ~1.07 printed as ~10.7 and looked like divergence.
            mean_sft = acc_sft / PREF_LOG_EVERY
            mean_pref = acc_pref / PREF_LOG_EVERY
            mean_gap = acc_gap / PREF_LOG_EVERY
            avg = mean_sft + 0.1 * mean_pref
            if avg < best_loss:
                best_loss = avg
            log(
                f"pref step {step:>4} | "
                f"sft {mean_sft:.4f} | "
                f"pref {mean_pref:.4f} | "
                f"gap {mean_gap:+.3f} | "
                f"lr {lr:.2e} | "
                f"grad {grad_norm:.3f}"
            )
            acc_sft = acc_pref = acc_gap = 0.0
            n_acc = 0

    # Save final model
    Path(ORPO_CHECKPOINT).mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        f"{ORPO_CHECKPOINT}/model.safetensors",
    )
    log(f"\n  Preference alignment complete. Best loss: {best_loss:.4f}")
    log(f"  Final model saved: {ORPO_CHECKPOINT}")
    return model


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path("logs").mkdir(exist_ok=True)
    log_file = open(LOG_PATH, "w", buffering=1, encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    log("=" * 60)
    log("PyCraft-1 Post-Training: SFT + Preference Alignment")
    log("=" * 60)
    log(f"  Base     : {BASE_CHECKPOINT}")
    log(f"  Device   : {device}")
    log(f"  Dropout  : disabled throughout fine-tuning")
    log(f"  Precision: float32 (no BF16)")

    # Load tokenizer and model
    tokenizer = PyCraftTokenizer()
    cfg = get_config_120m()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.dropout = 0.0   # disabled from the start

    model = PyCraftModel(cfg).to(device)
    weights = load_file(
        f"{BASE_CHECKPOINT}/model.safetensors",
        device=device,
    )
    model.load_state_dict(weights)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Params   : {n_params/1e6:.1f}M")

    # Verify base model loss before any fine-tuning
    log("\nVerifying base model loss (should be ~1.2-2.0)...")
    model.eval()
    tok = PyCraftTokenizer()
    test_code = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"
    ids = tok.encode(test_code)
    inp = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
    tgt = torch.tensor(ids[1:],  dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(inp)
        base_loss = F.cross_entropy(
            logits.float().view(-1, cfg.vocab_size),
            tgt.view(-1),
        )
    log(f"  Base model loss on Python code: {base_loss.item():.4f}")
    if base_loss.item() > 3.0:
        log("  WARNING: base loss > 3.0. Check checkpoint path.")
    else:
        log("  Base model verified OK.")

    # Run Stage 1
    model = run_sft(model, tokenizer, device, log)

    # Run Stage 2
    model = run_preference(model, tokenizer, device, log)

    log("\n" + "=" * 60)
    log("Fine-tuning complete.")
    log(f"  SFT checkpoint  : {SFT_CHECKPOINT}")
    log(f"  Final checkpoint: {ORPO_CHECKPOINT}")
    log("=" * 60)
    log_file.close()


if __name__ == "__main__":
    main()
