# model/feedforward.py
# SwiGLU Feed-Forward Network.
#
# Standard FFN uses: Linear → ReLU → Linear  (2 matrices)
# SwiGLU uses:       (Linear_gate ⊙ Swish(Linear_up)) → Linear_down  (3 matrices)
#
# The gating mechanism gives better accuracy per FLOP.
# Used in: Llama 3, Qwen 3, Gemma 2, Mistral, PyCraft-1.
# No bias terms throughout.

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import PyCraftConfig


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block.

    Architecture:
        x → [gate_proj(x) * silu(up_proj(x))] → down_proj → output

    Where silu(x) = x * sigmoid(x)  (also called Swish)
    """

    def __init__(self, config: PyCraftConfig):
        super().__init__()
        # Three projections, all without bias
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate branch: controls information flow
        gate = self.gate_proj(x)
        # Up branch: projects to higher dimension
        up = self.up_proj(x)
        # SwiGLU: element-wise gate * SiLU(up)
        # SiLU(x) = x * sigmoid(x) — smooth, non-monotonic activation
        hidden = gate * F.silu(up)
        # Project back to d_model
        return self.down_proj(hidden)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------ #
# Quick self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from model.config import get_config_tiny

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config_tiny()

    print(f"Testing SwiGLU FFN on {device}...")
    print(f"  d_model={cfg.d_model}, d_ff={cfg.d_ff}")

    ffn = SwiGLU(cfg).to(device)
    print(f"  FFN params: {ffn.param_count:,}")

    x = torch.randn(2, 64, cfg.d_model, device=device)
    with torch.no_grad():
        out = ffn(x)

    print(f"  Input  shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    assert out.shape == x.shape, "Output shape mismatch!"

    # Gradient test
    x2 = torch.randn(2, 64, cfg.d_model, device=device, requires_grad=True)
    out2 = ffn(x2)
    out2.sum().backward()
    assert x2.grad is not None
    print(f"  Gradient norm: {x2.grad.norm().item():.4f}")
    print("  All FFN tests PASSED.")
