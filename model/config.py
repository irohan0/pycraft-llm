# model/config.py
# PyCraft-1 model configuration
# All architectural hyperparameters live here.
# Other files import this — never hardcode numbers elsewhere.

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class PyCraftConfig:
    # ------------------------------------------------------------------ #
    # Vocabulary & sequence
    # ------------------------------------------------------------------ #
    vocab_size: int = 32000          # BPE tokenizer vocab (trained in Phase 2)
    max_seq_len: int = 2048          # context window

    # ------------------------------------------------------------------ #
    # Model dimensions
    # ------------------------------------------------------------------ #
    d_model: int = 512               # embedding / hidden dimension
    n_layers: int = 8                # number of transformer blocks
    n_heads: int = 8                 # number of query heads
    # number of key/value heads (GQA 4:1 ratio)
    n_kv_heads: int = 2

    # SwiGLU FFN intermediate dim.
    # Standard formula: (4 * d_model * 2/3), rounded to nearest multiple of 64
    # 512 * 4 * 2/3 = 1365.3 → round to 1408 for clean tensor ops
    d_ff: int = 1408

    # ------------------------------------------------------------------ #
    # Attention settings
    # ------------------------------------------------------------------ #
    use_qk_norm: bool = True         # QK-Norm (OLMo 2 / Qwen 3 technique)
    rope_theta: float = 10000.0      # RoPE base frequency
    attn_dropout: float = 0.0        # keep 0.0 during pretraining

    # ------------------------------------------------------------------ #
    # Training knobs
    # ------------------------------------------------------------------ #
    dropout: float = 0.0             # residual dropout (0 for pretraining)
    weight_tying: bool = False       # tie input embedding ↔ output projection
    # False: we have enough params at 120M

    # ------------------------------------------------------------------ #
    # FIM (Fill-in-the-Middle) special token IDs
    # These will be set after tokenizer training.
    # Defaults match standard FIM token positions.
    # ------------------------------------------------------------------ #
    fim_prefix_id: Optional[int] = None
    fim_suffix_id: Optional[int] = None
    fim_middle_id: Optional[int] = None
    fim_pad_id: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Derived / computed properties
    # ------------------------------------------------------------------ #
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.d_model // self.n_heads

    @property
    def n_heads_per_kv(self) -> int:
        """How many Q heads share each KV head."""
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        return self.n_heads // self.n_kv_heads

    @property
    def param_count_approx(self) -> int:
        """Rough parameter count for sanity checking."""
        embed = self.vocab_size * self.d_model
        attn = self.n_layers * (
            self.d_model * self.d_model +          # Wq
            2 * self.d_model * (self.n_kv_heads * self.head_dim) +  # Wk, Wv
            self.d_model * self.d_model            # Wo
        )
        ffn = self.n_layers * (
            3 * self.d_model * self.d_ff           # SwiGLU: gate, up, down
        )
        norms = self.n_layers * 2 * self.d_model  # RMSNorm per block x2
        lm_head = self.vocab_size * self.d_model   # output projection
        return embed + attn + ffn + norms + lm_head

    def __post_init__(self):
        # Validate GQA constraint
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads must be divisible by n_kv_heads. "
            f"Got {self.n_heads} and {self.n_kv_heads}."
        )
        # Validate head dimension
        assert self.d_model % self.n_heads == 0, (
            f"d_model must be divisible by n_heads."
        )


# ------------------------------------------------------------------ #
# Convenience presets
# ------------------------------------------------------------------ #

def get_config_120m() -> PyCraftConfig:
    """
    PyCraft-1 (120M parameters).
    Designed to train on a single RTX 3050 4GB laptop GPU.
    """
    return PyCraftConfig(
        vocab_size=32000,
        max_seq_len=1024,
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        d_ff=1408,
        use_qk_norm=True,
        rope_theta=10000.0,
        dropout=0.1,
    )


def get_config_tiny() -> PyCraftConfig:
    """
    PyCraft-tiny (~ 15M parameters).
    For rapid iteration and smoke-testing the training loop.
    Use this first before committing to a full training run.
    """
    return PyCraftConfig(
        vocab_size=32000,
        max_seq_len=512,
        d_model=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_ff=704,
        use_qk_norm=True,
        rope_theta=10000.0,
    )


# ------------------------------------------------------------------ #
# Quick self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    cfg = get_config_120m()
    params_m = cfg.param_count_approx / 1e6
    print(f"PyCraft-1 config loaded.")
    print(f"  d_model      : {cfg.d_model}")
    print(f"  n_layers     : {cfg.n_layers}")
    print(f"  n_heads      : {cfg.n_heads}  (Q)")
    print(
        f"  n_kv_heads   : {cfg.n_kv_heads}  (KV, GQA {cfg.n_heads_per_kv}:1)")
    print(f"  head_dim     : {cfg.head_dim}")
    print(f"  d_ff         : {cfg.d_ff}  (SwiGLU)")
    print(f"  QK-Norm      : {cfg.use_qk_norm}")
    print(f"  Approx params: {params_m:.1f}M")

    cfg_tiny = get_config_tiny()
    print(f"\nPyCraft-tiny config loaded.")
    print(f"  Approx params: {cfg_tiny.param_count_approx / 1e6:.1f}M")
