# model/attention.py
# Grouped-Query Attention with:
#   - Rotary Positional Embeddings (RoPE)
#   - QK-Norm (2025 technique from OLMo 2 / Qwen 3)
#   - PyTorch native SDPA (backend auto-selected, no flash-attn dependency)
#   - Optional KV-cache for incremental decoding
#   - No bias terms (standard in modern LLMs)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import PyCraftConfig


# ------------------------------------------------------------------ #
# RMSNorm
# Faster than LayerNorm: no mean subtraction, just RMS scaling.
# Used in: Llama 3, Qwen 3, Gemma 2, Mistral, PyCraft-1.
# ------------------------------------------------------------------ #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learnable scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        # Compute RMS over last dimension, then scale
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ------------------------------------------------------------------ #
# Rotary Positional Embeddings (RoPE)
# Encodes relative position by rotating Q and K vectors.
# No learned parameters — positional info is injected at runtime.
# ------------------------------------------------------------------ #
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        # Compute inverse frequencies for each pair of dimensions
        # Shape: (head_dim // 2,)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache for all positions up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        # Outer product: (seq_len, head_dim // 2)
        freqs = torch.outer(positions, self.inv_freq)
        # Duplicate for full head_dim: (seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension to implement complex multiply."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,   # (batch, n_heads, seq_len, head_dim)
        k: torch.Tensor,   # (batch, n_kv_heads, seq_len, head_dim)
        offset: int = 0,   # absolute position of the first token
    ):
        seq_len = q.shape[2]
        max_pos = self.cos_cache.shape[0]

        # Slicing past the end returns a SHORT tensor instead of raising,
        # which surfaces later as a confusing broadcast error — or worse, as
        # a plausible but positionally wrong result. Fail loudly instead.
        if offset < 0 or offset + seq_len > max_pos:
            raise ValueError(
                f"RoPE positions [{offset}, {offset + seq_len}) exceed the "
                f"precomputed table of {max_pos}. PyCraft-1 was trained with "
                f"max_seq_len={max_pos}; truncate the prompt or stop generating."
            )

        # .to() is a no-op when dtypes already match; _build_cache hardcodes
        # fp32, so this guards half/bf16 models.
        cos = self.cos_cache[offset: offset + seq_len].to(q.dtype)
        sin = self.sin_cache[offset: offset + seq_len].to(q.dtype)

        # Reshape for broadcasting: (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ------------------------------------------------------------------ #
# Grouped-Query Attention (GQA)
# n_kv_heads < n_heads: each KV head is shared by (n_heads // n_kv_heads) Q heads.
# Reduces KV-cache size by 4x with negligible quality loss.
# ------------------------------------------------------------------ #
class GroupedQueryAttention(nn.Module):
    def __init__(self, config: PyCraftConfig, layer_idx: int = 0):
        super().__init__()
        # Plain int — never enters state_dict, so checkpoints stay compatible.
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.use_qk_norm = config.use_qk_norm
        self.n_rep = config.n_heads_per_kv   # Q heads per KV head

        # Projections — no bias (standard in modern LLMs)
        self.wq = nn.Linear(config.d_model, config.n_heads *
                            config.head_dim, bias=False)
        self.wk = nn.Linear(
            config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(
            config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim,
                            config.d_model,    bias=False)

        # QK-Norm: RMSNorm on Q and K before RoPE
        # From OLMo 2 (2025) and Qwen 3 — stabilises training of small models
        if self.use_qk_norm:
            self.q_norm = RMSNorm(config.head_dim)
            self.k_norm = RMSNorm(config.head_dim)

        # RoPE
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to match Q head count for SDPA.
        (batch, n_kv_heads, seq, head_dim) → (batch, n_heads, seq, head_dim)
        """
        if self.n_rep == 1:
            return x
        batch, n_kv, seq, head_dim = x.shape
        x = x.unsqueeze(2).expand(batch, n_kv, self.n_rep, seq, head_dim)
        return x.reshape(batch, n_kv * self.n_rep, seq, head_dim)

    def forward(
        self,
        x: torch.Tensor,                         # (batch, seq_len, d_model)
        attn_mask: torch.Tensor | None = None,   # (1|B, 1, T, S) bool, True = attend
        kv_cache=None,                           # KVCache, or None to disable
        position_offset: int = 0,                # cached positions before this call
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # 1. Project to Q, K, V
        q = self.wq(x)   # (batch, seq, n_heads * head_dim)
        k = self.wk(x)   # (batch, seq, n_kv_heads * head_dim)
        v = self.wv(x)   # (batch, seq, n_kv_heads * head_dim)

        # 2. Reshape into (batch, heads, seq, head_dim)
        q = q.view(batch, seq_len, self.n_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads,
                   self.head_dim).transpose(1, 2)

        # 3. QK-Norm (applied per-head, before RoPE)
        #    Normalise each head vector independently.
        #    Cache-safe: RMSNorm reduces over the last dim only, so position t
        #    depends on nothing but the vector at position t.
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 4. Apply RoPE at true absolute positions
        q, k = self.rope(q, k, offset=position_offset)

        # 5. Append to the cache, read back everything written so far.
        #    Stored PRE-expansion (n_kv_heads) — that 4x is the GQA memory win.
        #    Keys are cached POST-RoPE; never re-rotate them.
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        # 6. Expand K and V to match Q head count (GQA expansion)
        k = self._repeat_kv(k)   # (batch, n_heads, kv_len, head_dim)
        v = self._repeat_kv(v)

        # 7. Scaled dot-product attention.
        #    Backend is left to PyTorch: pinning one breaks CPU (the
        #    memory-efficient kernel is CUDA-only) and blocks flash on GPU.
        #
        #    is_causal MUST be conditional. PyTorch builds it upper-left
        #    aligned, so on a decode step (q_len=1, kv_len=N) it would mask
        #    everything but the first prompt token — silently, with no error.
        #    See build_attn_mask() in model/kv_cache.py.
        kv_len = k.shape[2]
        if attn_mask is None:
            # Square (offset == 0) -> upper-left causal is correct.
            # Single query (q_len == 1) -> may attend to all cached keys.
            is_causal = (seq_len == kv_len)
        else:
            is_causal = False   # SDPA rejects mask + is_causal together

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        # 8. Merge heads and project back to d_model
        # (batch, n_heads, seq, head_dim) → (batch, seq, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model)
        return self.wo(attn_out)


# ------------------------------------------------------------------ #
# Quick self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from model.config import get_config_tiny

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config_tiny()

    print(f"Testing GroupedQueryAttention on {device}...")
    print(f"  n_heads={cfg.n_heads}, n_kv_heads={cfg.n_kv_heads}, "
          f"head_dim={cfg.head_dim}, use_qk_norm={cfg.use_qk_norm}")

    attn = GroupedQueryAttention(cfg).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in attn.parameters())
    print(f"  Attention block params: {n_params:,}")

    # Forward pass
    x = torch.randn(2, 64, cfg.d_model, device=device)  # batch=2, seq=64
    with torch.no_grad():
        out = attn(x)

    print(f"  Input  shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    assert out.shape == x.shape, "Output shape mismatch!"

    # Test gradient flow
    x.requires_grad_(True)
    x_grad = torch.randn(2, 64, cfg.d_model, device=device)
    x_grad.requires_grad_(True)
    out2 = attn(x_grad)
    loss = out2.sum()
    loss.backward()
    assert x_grad.grad is not None, "No gradient!"
    print(f"  Gradient norm: {x_grad.grad.norm().item():.4f}")
    print("  All attention tests PASSED.")
