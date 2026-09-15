# model/kv_cache.py
#
# KV-cache for PyCraft-1 incremental decoding.
#
# Without a cache, generating token N re-runs the full forward pass over all
# N-1 previous tokens — quadratic work for a linear amount of output.
# This module stores each layer's keys and values so every decode step only
# computes the single newest position.
#
# Two design choices matter and are easy to get wrong:
#
#   1. Storage is PRE-expansion — (B, n_kv_heads, ...) not (B, n_heads, ...).
#      With GQA 8Q/2KV that is 4x less memory, which is the entire point of
#      grouped-query attention. _repeat_kv re-expands on read.
#
#   2. Memory is PREALLOCATED, never torch.cat'd. Concatenating per step costs
#      O(N) copy per step and O(N^2) overall — roughly 4.3 GB of pointless
#      memcpy across one full-context generation.
#
# The cache holds POST-RoPE keys. RoPE is relative (q_m . k_n depends only on
# m - n), so a key rotated once at its true absolute position stays valid
# forever. Never re-apply RoPE to cached keys.

import torch


# ------------------------------------------------------------------ #
# KV cache
# ------------------------------------------------------------------ #
class KVCache:
    """
    Pre-allocated per-layer key/value store for incremental decoding.

    Per layer:   k[i], v[i] : (batch, n_kv_heads, max_len, head_dim)
    Live region: [:, :, :seq_len, :]

    Typical use:

        cache = KVCache.from_model(model, batch_size=1)
        logits, _ = model(prompt_ids, past_key_values=cache)   # prefill
        logits, _ = model(next_id,    past_key_values=cache)   # decode
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        max_len: int,
        device,
        dtype: torch.dtype = torch.float32,
    ):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.device = device
        self.dtype = dtype
        self.seq_len = 0

        shape = (batch_size, n_kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(n_layers)]

    # -------------------------------------------------------------- #
    # Called once per layer, per forward pass
    # -------------------------------------------------------------- #
    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,   # (batch, n_kv_heads, T, head_dim)
        v_new: torch.Tensor,   # (batch, n_kv_heads, T, head_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Write this layer's new K/V into the cache and return views covering
        everything written so far: (batch, n_kv_heads, seq_len + T, head_dim).

        Deliberately does NOT advance seq_len. The model advances it once,
        after all layers have written. Advancing here would make layer i see
        the offset that only layer i+1 should see — a silent corruption that
        produces plausible-looking garbage.
        """
        batch, _, T, _ = k_new.shape
        start = self.seq_len

        if start + T > self.max_len:
            raise ValueError(
                f"KVCache overflow: {start} + {T} > max_len={self.max_len}. "
                f"Allocate a larger cache or stop generating."
            )
        if batch != self.batch_size:
            raise ValueError(
                f"batch {batch} does not match cache batch {self.batch_size}"
            )
        if k_new.dtype != self.dtype:
            # Silent casting here would quietly degrade precision every step.
            raise TypeError(
                f"cache dtype {self.dtype} != incoming dtype {k_new.dtype}"
            )

        self.k[layer_idx][:, :, start:start + T] = k_new
        self.v[layer_idx][:, :, start:start + T] = v_new

        # Narrowed views — the zero-padded tail is never visible, so reset()
        # needs no memset.
        return (
            self.k[layer_idx][:, :, :start + T],
            self.v[layer_idx][:, :, :start + T],
        )

    def advance(self, n: int):
        """Advance the write head. Called once per forward, by the model."""
        self.seq_len += n

    def reset(self):
        """Reuse this cache for a new sequence. No zeroing required."""
        self.seq_len = 0

    def __len__(self) -> int:
        return self.seq_len

    def __repr__(self) -> str:
        return (
            f"KVCache(layers={self.n_layers}, batch={self.batch_size}, "
            f"kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
            f"seq_len={self.seq_len}/{self.max_len}, dtype={self.dtype})"
        )

    @property
    def memory_bytes(self) -> int:
        """Total allocated cache size in bytes (K and V, all layers)."""
        per = (self.batch_size * self.n_kv_heads
               * self.max_len * self.head_dim)
        return 2 * self.n_layers * per * torch.empty(
            (), dtype=self.dtype).element_size()

    # -------------------------------------------------------------- #
    @classmethod
    def from_model(
        cls,
        model,
        batch_size: int = 1,
        max_len: int | None = None,
        device=None,
        dtype: torch.dtype | None = None,
    ) -> "KVCache":
        """Build a cache matching a model's config, device, and dtype."""
        cfg = model.config
        # Read device/dtype from the embedding, not a Linear: dynamic
        # quantization replaces nn.Linear with a module whose .weight is a
        # method. Activations stay fp32 under dynamic quant either way.
        ref = model.token_embedding.weight
        return cls(
            n_layers=cfg.n_layers,
            batch_size=batch_size,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            max_len=max_len or cfg.max_seq_len,
            device=device if device is not None else ref.device,
            dtype=dtype if dtype is not None else ref.dtype,
        )


# ------------------------------------------------------------------ #
# Causal mask construction
# ------------------------------------------------------------------ #
def build_attn_mask(
    q_len: int,
    offset: int,
    device,
) -> torch.Tensor | None:
    """
    Build a bottom-right-aligned causal mask for cached attention.

    Returns None for the two cases SDPA handles without an explicit mask:

        offset == 0   caller passes is_causal=True (square, so PyTorch's
                      upper-left alignment is already correct)
        q_len  == 1   the single newest query may attend to every cached
                      key, so causality holds structurally — no mask needed

    Otherwise returns (1, 1, q_len, offset + q_len) bool, True = "may attend":

        mask[i, j] = (j <= offset + i)

    WHY THIS EXISTS: PyTorch builds is_causal as torch.ones(L, S).tril() —
    UPPER-LEFT aligned. With L=1, S=N that mask contains exactly one True
    (column 0), so a cached decode step with is_causal=True would attend only
    to the first prompt token, at every layer. No error, no NaN — just output
    that ignores the prompt and collapses into repetition. Hence the explicit
    dispatch rather than a blanket is_causal=True.
    """
    if offset == 0 or q_len == 1:
        return None

    kv_len = offset + q_len
    q_pos = torch.arange(offset, kv_len, device=device).unsqueeze(1)  # (T, 1)
    k_pos = torch.arange(0, kv_len, device=device).unsqueeze(0)       # (1, S)
    return (k_pos <= q_pos)[None, None, :, :]                        # (1,1,T,S)


# ------------------------------------------------------------------ #
# Quick self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Testing build_attn_mask...")

    # Fast paths return None
    assert build_attn_mask(4, 0, "cpu") is None, "offset=0 should return None"
    assert build_attn_mask(1, 7, "cpu") is None, "q_len=1 should return None"

    # Explicit mask matches a brute-force construction
    for q_len, offset in [(4, 7), (3, 1), (2, 2)]:
        got = build_attn_mask(q_len, offset, "cpu")
        kv_len = offset + q_len
        assert got.shape == (1, 1, q_len, kv_len), f"bad shape {got.shape}"
        for i in range(q_len):
            for j in range(kv_len):
                expected = (j <= offset + i)
                assert bool(got[0, 0, i, j]) == expected, (
                    f"mask[{i}][{j}] wrong for q_len={q_len}, offset={offset}"
                )
    print("  build_attn_mask: OK")

    print("\nTesting KVCache...")
    cache = KVCache(n_layers=2, batch_size=1, n_kv_heads=2,
                    head_dim=64, max_len=16, device="cpu")
    print(f"  {cache}")
    print(f"  allocated: {cache.memory_bytes / 1024:.1f} KiB")

    # Prefill 4 positions across both layers
    k = torch.randn(1, 2, 4, 64)
    v = torch.randn(1, 2, 4, 64)
    for layer in range(2):
        kk, vv = cache.update(layer, k, v)
        assert kk.shape == (1, 2, 4, 64), f"bad view shape {kk.shape}"
    assert cache.seq_len == 0, "update() must not advance seq_len"
    cache.advance(4)
    assert len(cache) == 4

    # One decode step
    k1 = torch.randn(1, 2, 1, 64)
    v1 = torch.randn(1, 2, 1, 64)
    kk, vv = cache.update(0, k1, v1)
    assert kk.shape == (1, 2, 5, 64), f"bad decode view {kk.shape}"
    assert torch.equal(kk[:, :, :4], k), "prefill K was corrupted"
    assert torch.equal(kk[:, :, 4:], k1), "decode K not written"
    cache.advance(1)
    print("  writes and views: OK")

    # Guards
    for bad, exc, label in [
        (lambda: cache.update(0, torch.randn(1, 2, 99, 64),
                              torch.randn(1, 2, 99, 64)), ValueError, "overflow"),
        (lambda: cache.update(0, torch.randn(1, 2, 1, 64).half(),
                              torch.randn(1, 2, 1, 64).half()), TypeError, "dtype"),
        (lambda: cache.update(0, torch.randn(3, 2, 1, 64),
                              torch.randn(3, 2, 1, 64)), ValueError, "batch"),
    ]:
        try:
            bad()
            raise AssertionError(f"{label} guard did not fire")
        except exc:
            pass
    print("  guards: OK")

    cache.reset()
    assert len(cache) == 0
    print("  reset: OK")
    print("\nAll kv_cache tests PASSED.")
