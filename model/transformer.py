# model/transformer.py
# A single PyCraft-1 transformer block.
#
# Layout (pre-norm, standard in all 2025 LLMs):
#   x → RMSNorm → GQA → residual add
#     → RMSNorm → SwiGLU → residual add
#
# Pre-norm (normalise before the sublayer) is more stable than
# post-norm during early training — critical for small models.

import torch
import torch.nn as nn

from model.config import PyCraftConfig
from model.attention import GroupedQueryAttention, RMSNorm
from model.feedforward import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, config: PyCraftConfig):
        super().__init__()
        # Pre-norm for attention sublayer
        self.norm1 = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)

        # Pre-norm for FFN sublayer
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,                       # (batch, seq_len, d_model)
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention sublayer with residual
        x = x + self.attn(self.norm1(x), attn_mask)
        # FFN sublayer with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ------------------------------------------------------------------ #
# Quick self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from model.config import get_config_tiny

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config_tiny()

    print(f"Testing TransformerBlock on {device}...")
    block = TransformerBlock(cfg).to(device)

    n_params = sum(p.numel() for p in block.parameters())
    print(f"  Block params: {n_params:,}")

    x = torch.randn(2, 64, cfg.d_model, device=device)
    with torch.no_grad():
        out = block(x)

    print(f"  Input  shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    assert out.shape == x.shape

    # Gradient test
    x2 = torch.randn(2, 64, cfg.d_model, device=device, requires_grad=True)
    block(x2).sum().backward()
    assert x2.grad is not None
    print(f"  Gradient norm: {x2.grad.norm().item():.4f}")
    print("  TransformerBlock test PASSED.")
