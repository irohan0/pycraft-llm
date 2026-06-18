# training/train.py
#
# Entry point for PyCraft-1 pretraining.
# Run with:  python -m training.train
#
# This wires together:
#   model      → PyCraftModel (55M params, RTX 3050 optimised)
#   tokenizer  → trained BPE (32k vocab, Python-tuned)
#   dataset    → codeparrot/github-code streamed Python
#   trainer    → BF16 + grad accumulation + cosine LR + checkpointing

import torch
from torch.utils.data import DataLoader

from model.pycraft_model import PyCraftModel
from model.config import get_config_120m, get_config_tiny
from tokenizer.tokenizer_utils import PyCraftTokenizer
from data.stream_dataset import PythonCodeStreamDataset
from training.trainer import PyCraftTrainer, TrainerConfig


def main():
    # ---------------------------------------------------------------- #
    # 0. Device setup
    # ---------------------------------------------------------------- #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # ---------------------------------------------------------------- #
    # 1. Load tokenizer
    # ---------------------------------------------------------------- #
    print("\nLoading tokenizer...")
    tokenizer = PyCraftTokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  EOT id: {tokenizer.eot_id}")

    # ---------------------------------------------------------------- #
    # 2. Model
    #
    # IMPORTANT: Start with get_config_tiny() for your first run.
    # This lets you verify the full pipeline works end-to-end in minutes.
    # Once confirmed, switch to get_config_120m() for real training.
    # ---------------------------------------------------------------- #
    print("\nBuilding model...")

    USE_TINY = False   # ← Set to False when ready for full training

    if USE_TINY:
        print("  Using TINY config (~15M params) for pipeline verification.")
        print("  Change USE_TINY = False in train.py for full training.")
        cfg = get_config_tiny()
        cfg.vocab_size = tokenizer.vocab_size   # match trained tokenizer
    else:
        print("  Using FULL 120M config.")
        cfg = get_config_120m()
        cfg.vocab_size = tokenizer.vocab_size

    # Set FIM token IDs from the trained tokenizer
    cfg.fim_prefix_id = tokenizer.prefix_id
    cfg.fim_suffix_id = tokenizer.suffix_id
    cfg.fim_middle_id = tokenizer.middle_id

    model = PyCraftModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.2f}M")

    # Compile model for faster training (PyTorch 2.x)
    # Skipped on Windows because torch.compile requires Triton which
    # is Linux-only. We get good speed from BF16 + fused AdamW instead.
    # model = torch.compile(model)  # uncomment if on Linux

    # ---------------------------------------------------------------- #
    # 3. Dataset + DataLoader
    # ---------------------------------------------------------------- #
    print("\nSetting up data pipeline...")
    dataset = PythonCodeStreamDataset(
        tokenizer=tokenizer,
        seq_len=cfg.max_seq_len,
        fim_rate=0.5,
        seed=42,
    )

    # num_workers=0 on Windows (multiprocessing with IterableDataset
    # causes issues on Windows — single-process data loading is fine)
    loader = DataLoader(
        dataset,
        batch_size=4,          # micro batch size
        num_workers=0,         # must be 0 on Windows with streaming datasets
        pin_memory=True,
        prefetch_factor=None,  # only valid when num_workers > 0
    )
    print("  Data loader ready (streaming, no download required)")

    # ---------------------------------------------------------------- #
    # 4. Trainer config
    #
    # Tiny run: 500 steps to verify pipeline (~5 minutes)
    # Full run: 100,000 steps (~several nights of training)
    # ---------------------------------------------------------------- #
    trainer_cfg = TrainerConfig(
        checkpoint_dir="checkpoints",
        log_dir="logs",
        learning_rate=3e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        micro_batch_size=4,
        gradient_accumulation_steps=64,
        warmup_steps=500,
        max_steps=4000,
        log_every=10,
        checkpoint_every=200,
        keep_checkpoints=5,
        resume=True,
    )

    print(
        f"\n  Effective batch size : {trainer_cfg.effective_batch_size} sequences")
    print(f"  Max steps            : {trainer_cfg.max_steps:,}")
    print(f"  Warmup steps         : {trainer_cfg.warmup_steps:,}")

    # ---------------------------------------------------------------- #
    # 5. Train
    # ---------------------------------------------------------------- #
    trainer = PyCraftTrainer(
        model=model,
        train_loader=loader,
        config=trainer_cfg,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
