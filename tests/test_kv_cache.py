# tests/test_kv_cache.py
#
# Correctness tests for PyCraft-1's KV cache and sampling.
#
#   python -m tests.test_kv_cache
#
# Plain asserts, no pytest (it is not in the environment).
#
# The invariant under test: in an uncached forward, the K/V at position t in
# layer l is a function of hidden state h_l[t], which by causality depends
# only on positions <= t. In the cached path that same K/V was computed when
# t was the newest token, from the identical hidden state. The two must agree.
#
# A mask bug during prefill poisons the cached K/V for every later step, which
# is why chunked prefill is tested and not just single-token decode.

import json
import math
import struct
import time
from pathlib import Path

import torch

from model.config import get_config_tiny, get_config_120m
from model.kv_cache import KVCache, build_attn_mask
from model.pycraft_model import PyCraftModel

SEED = 1234
CHECKPOINT = Path("checkpoints/sft_stage1/model.safetensors")

_passed = 0
_skipped = 0


def ok(label: str):
    global _passed
    _passed += 1
    print(f"  PASS  {label}")


def skip(label: str, why: str):
    global _skipped
    _skipped += 1
    print(f"  SKIP  {label}  ({why})")


def build_model(cfg=None):
    torch.manual_seed(SEED)
    cfg = cfg or get_config_tiny()
    model = PyCraftModel(cfg).to("cpu").float()
    model.eval()
    return model, cfg


# ------------------------------------------------------------------ #
# 1. Checkpoint compatibility — the constraint gate
# ------------------------------------------------------------------ #
def test_checkpoint_keys():
    """
    Reads the safetensors HEADER ONLY (8-byte LE length + JSON) so this costs
    nothing. Catches an accidentally-persistent buffer, which would break
    strict=True loading in eval/ and training/.
    """
    if not CHECKPOINT.exists():
        skip("checkpoint key set", f"{CHECKPOINT} not present")
        return

    with open(CHECKPOINT, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    ckpt_keys = set(header) - {"__metadata__"}

    tok_vocab = 32000
    cfg = get_config_120m()
    cfg.vocab_size = tok_vocab
    cfg.dropout = 0.0
    model, _ = build_model(cfg)
    model_keys = set(model.state_dict().keys())

    missing = ckpt_keys - model_keys
    extra = model_keys - ckpt_keys
    assert not missing, f"checkpoint has keys the model lacks: {sorted(missing)}"
    assert not extra, f"model gained keys not in checkpoint: {sorted(extra)}"
    assert len(ckpt_keys) == 91, f"expected 91 tensors, found {len(ckpt_keys)}"
    ok(f"checkpoint key set ({len(ckpt_keys)} tensors, strict=True safe)")


# ------------------------------------------------------------------ #
# 2. Per-position logit equivalence — the core proof
# ------------------------------------------------------------------ #
def test_logit_equivalence():
    model, cfg = build_model()
    T, prefill = 48, 16
    torch.manual_seed(SEED)
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        ref, _ = model(ids)                      # (1, T, V) uncached reference

    cache = KVCache.from_model(model, batch_size=1, max_len=T)
    with torch.no_grad():
        got, _ = model(ids[:, :prefill], past_key_values=cache)

    # Prefill block must match exactly — this is also the cache-poisoning check
    torch.testing.assert_close(got, ref[:, :prefill], rtol=1e-4, atol=1e-4)

    max_prob_delta = 0.0
    for t in range(prefill, T):
        with torch.no_grad():
            step, _ = model(ids[:, t:t + 1], past_key_values=cache,
                            num_logits_to_keep=1)
        got_t = step[:, -1, :]
        ref_t = ref[:, t, :]

        torch.testing.assert_close(got_t, ref_t, rtol=1e-4, atol=1e-4)
        # The property that actually decides generation
        assert int(got_t.argmax()) == int(ref_t.argmax()), (
            f"argmax diverged at position {t}"
        )
        delta = (got_t.softmax(-1) - ref_t.softmax(-1)).abs().max().item()
        max_prob_delta = max(max_prob_delta, delta)

    assert max_prob_delta < 1e-5, f"prob delta {max_prob_delta:.2e} too large"
    assert len(cache) == T, f"cache seq_len {len(cache)} != {T}"
    ok(f"per-position logits, 100% argmax match "
       f"(max prob delta {max_prob_delta:.2e})")


# ------------------------------------------------------------------ #
# 3. Chunked prefill — the only test exercising offset>0 with q_len>1
# ------------------------------------------------------------------ #
def test_chunked_prefill():
    model, cfg = build_model()
    chunks = [16, 8, 1, 23]
    T = sum(chunks)
    torch.manual_seed(SEED)
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        ref, _ = model(ids)

    cache = KVCache.from_model(model, batch_size=1, max_len=T)
    pos = 0
    for n in chunks:
        with torch.no_grad():
            got, _ = model(ids[:, pos:pos + n], past_key_values=cache)
        torch.testing.assert_close(
            got, ref[:, pos:pos + n], rtol=1e-4, atol=1e-4)
        pos += n
    ok(f"chunked prefill {chunks} matches single pass")


# ------------------------------------------------------------------ #
# 4. Greedy generation equivalence — strongest end-to-end check
# ------------------------------------------------------------------ #
def test_greedy_equivalence():
    model, cfg = build_model()
    torch.manual_seed(SEED)
    prompt = torch.randint(0, cfg.vocab_size, (1, 12))

    kw = dict(max_new_tokens=32, temperature=0.0, eos_token_id=None)
    with torch.no_grad():
        cached = model.generate(prompt, use_cache=True, **kw)
        uncached = model.generate(prompt, use_cache=False, **kw)

    if not torch.equal(cached, uncached):
        diff = (cached != uncached).nonzero()[0, 1].item()
        raise AssertionError(
            f"greedy paths diverged at index {diff}: "
            f"cached={cached[0, diff]} uncached={uncached[0, diff]}"
        )
    assert cached.shape[1] == 12 + 32
    ok("greedy generation: cached == uncached, token for token")


def test_seeded_equivalence():
    model, cfg = build_model()
    torch.manual_seed(SEED)
    prompt = torch.randint(0, cfg.vocab_size, (1, 12))

    kw = dict(max_new_tokens=24, temperature=0.8, top_k=50,
              eos_token_id=None, seed=0)
    with torch.no_grad():
        a = model.generate(prompt, use_cache=True, **kw)
        b = model.generate(prompt, use_cache=False, **kw)
    assert torch.equal(a, b), "seeded sampling diverged between cache paths"
    ok("seeded sampling: cached == uncached")


# ------------------------------------------------------------------ #
# 5. The is_causal regression guard
# ------------------------------------------------------------------ #
def test_is_causal_guard():
    """
    Proves the bug this design exists to avoid is actually detectable.

    PyTorch aligns is_causal to the upper-left, so at q_len=1, kv_len=N the
    mask keeps exactly one column — the model would attend only to the first
    prompt token. If this assert ever fails, that footgun has been
    reintroduced upstream and every other test here has gone blind.
    """
    torch.manual_seed(SEED)
    q = torch.randn(1, 4, 1, 64)     # single decode query
    k = torch.randn(1, 4, 20, 64)
    v = torch.randn(1, 4, 20, 64)

    correct = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=False)
    buggy = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=True)

    delta = (correct - buggy).abs().max().item()
    assert delta > 1e-2, (
        f"is_causal=True is indistinguishable from the correct path "
        f"(delta {delta:.2e}) — the regression guard is no longer meaningful"
    )
    # And confirm the buggy path really is "attend to token 0 only"
    torch.testing.assert_close(buggy, v[:, :, :1, :].expand_as(buggy),
                               rtol=1e-5, atol=1e-5)
    ok(f"is_causal guard: decode-with-is_causal collapses to token 0 "
       f"(delta {delta:.3f})")


# ------------------------------------------------------------------ #
# 6. Mask construction
# ------------------------------------------------------------------ #
def test_mask():
    assert build_attn_mask(4, 0, "cpu") is None, "offset=0 must return None"
    assert build_attn_mask(1, 7, "cpu") is None, "q_len=1 must return None"

    for q_len, offset in [(4, 7), (3, 1), (2, 2), (5, 11)]:
        got = build_attn_mask(q_len, offset, "cpu")
        kv_len = offset + q_len
        assert got.shape == (1, 1, q_len, kv_len)
        for i in range(q_len):
            for j in range(kv_len):
                assert bool(got[0, 0, i, j]) == (j <= offset + i), (
                    f"mask[{i}][{j}] wrong at q_len={q_len}, offset={offset}")
    ok("build_attn_mask matches brute-force construction")


# ------------------------------------------------------------------ #
# 7. RoPE offset identity
# ------------------------------------------------------------------ #
def test_rope_offset():
    model, cfg = build_model()
    rope = model.blocks[0].attn.rope
    torch.manual_seed(SEED)
    q = torch.randn(1, cfg.n_heads, 24, cfg.head_dim)
    k = torch.randn(1, cfg.n_kv_heads, 24, cfg.head_dim)

    q_full, k_full = rope(q, k, offset=0)
    for t in (0, 1, 7, 23):
        q_one, k_one = rope(q[:, :, t:t + 1], k[:, :, t:t + 1], offset=t)
        torch.testing.assert_close(q_one, q_full[:, :, t:t + 1],
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(k_one, k_full[:, :, t:t + 1],
                                   rtol=1e-6, atol=1e-6)

    # Past the table end must raise, not silently return a short tensor
    for offset in (cfg.max_seq_len, cfg.max_seq_len - 10, -1):
        try:
            rope(q, k, offset=offset)
            raise AssertionError(f"offset={offset} should have raised")
        except ValueError:
            pass
    ok("RoPE offset slicing is position-exact and bounds-checked")


# ------------------------------------------------------------------ #
# 8. Training contract unchanged
# ------------------------------------------------------------------ #
def test_training_contract():
    model, cfg = build_model()
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))

    logits, loss = model(ids, targets)
    assert logits.shape == (2, 32, cfg.vocab_size), f"bad shape {logits.shape}"
    assert loss is not None and torch.isfinite(loss)
    # Random init over a 32k vocab should land near ln(32000) ~= 10.37
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 1.0, (
        f"loss {loss.item():.3f} implausible for random init")
    loss.backward()
    assert model.lm_head.weight.grad is not None, "no gradient reached lm_head"

    logits_only, none_loss = model(ids)
    assert none_loss is None, "loss must be None without targets"

    # Masked labels must be honoured (the -1 vs -100 fix)
    masked = targets.clone()
    masked[:, :16] = -100
    _, masked_loss = model(ids, masked)
    assert torch.isfinite(masked_loss), "ignore_index=-100 not handled"

    assert model.training, "forward must not change training mode"
    ok("training contract: (logits, loss), backward, ignore_index=-100")


def test_generate_restores_mode():
    model, cfg = build_model()
    model.train()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    model.generate(prompt, max_new_tokens=4, eos_token_id=None)
    assert model.training, "generate() left the model in eval mode"
    ok("generate() restores training mode")


# ------------------------------------------------------------------ #
# 9. Cache reuse and guards
# ------------------------------------------------------------------ #
def test_cache_reuse():
    model, cfg = build_model()
    torch.manual_seed(SEED)
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    cache = KVCache.from_model(model, batch_size=1, max_len=64)

    kw = dict(max_new_tokens=16, temperature=0.0, eos_token_id=None)
    a = model.generate(prompt, past_key_values=cache, **kw)
    b = model.generate(prompt, past_key_values=cache, **kw)
    assert torch.equal(a, b), "reusing a cache changed the output (missing reset?)"
    ok("cache reuse across generate() calls is clean")


def test_eos_stops():
    model, cfg = build_model()
    torch.manual_seed(SEED)
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))

    # Forcing every token to be EOS must stop after exactly one
    out = model.generate(prompt, max_new_tokens=50, temperature=0.0,
                         eos_token_id=list(range(cfg.vocab_size)))
    assert out.shape[1] == 9, f"EOS did not stop generation (len {out.shape[1]})"

    # eos_token_id=None must run the full budget
    out = model.generate(prompt, max_new_tokens=12, temperature=0.0,
                         eos_token_id=None)
    assert out.shape[1] == 8 + 12
    ok("EOS stopping halts generation; None runs full budget")


def test_context_limit():
    model, cfg = build_model()
    prompt = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len - 4))
    out = model.generate(prompt, max_new_tokens=100, temperature=0.0,
                         eos_token_id=None)
    assert out.shape[1] <= cfg.max_seq_len, (
        f"generated past max_seq_len: {out.shape[1]} > {cfg.max_seq_len}")
    ok(f"hard stop at max_seq_len ({out.shape[1]} <= {cfg.max_seq_len})")


# ------------------------------------------------------------------ #
# 10. Batched generation
# ------------------------------------------------------------------ #
def test_batched_equivalence():
    """
    The load-bearing test for left padding. If the key-padding mask leaked, or
    the RoPE offset shift mattered, shorter (more heavily padded) rows would
    diverge from their single-sequence result while longer ones stayed fine.
    """
    model, cfg = build_model()
    torch.manual_seed(7)
    prompts = [
        torch.randint(0, cfg.vocab_size, (11,)).tolist(),
        torch.randint(0, cfg.vocab_size, (5,)).tolist(),   # most padding
        torch.randint(0, cfg.vocab_size, (18,)).tolist(),  # none
    ]
    kw = dict(temperature=0.0, eos_token_id=None)

    single = [
        model.generate(torch.tensor([p]), max_new_tokens=24, **kw)[0, len(p):].tolist()
        for p in prompts
    ]
    batched = model.generate_batch(prompts, max_new_tokens=24, **kw)

    for i, (s, b) in enumerate(zip(single, batched)):
        if s != b:
            d = next(j for j, (x, y) in enumerate(zip(s, b)) if x != y)
            raise AssertionError(
                f"row {i} (prompt len {len(prompts[i])}) diverged at token {d}: "
                f"single={s[d]} batched={b[d]}"
            )
    ok("batched generation matches single-sequence, row for row")


def test_batched_per_row_eos():
    model, cfg = build_model()
    torch.manual_seed(7)
    prompts = [torch.randint(0, cfg.vocab_size, (n,)).tolist() for n in (6, 9)]

    # Force EOS on every token: each row must stop immediately with 0 output
    out = model.generate_batch(prompts, max_new_tokens=20, temperature=0.0,
                               eos_token_id=list(range(cfg.vocab_size)))
    assert all(len(o) == 0 for o in out), f"EOS not honoured per row: {[len(o) for o in out]}"

    # eos_token_id=None runs the full budget for every row
    out = model.generate_batch(prompts, max_new_tokens=12, temperature=0.0,
                               eos_token_id=None)
    assert all(len(o) == 12 for o in out), f"budget not honoured: {[len(o) for o in out]}"

    assert model.generate_batch([], max_new_tokens=8) == [], "empty batch"
    ok("batched EOS truncates per row; empty batch handled")


def test_padding_mask():
    """Padded keys must be excluded, and no query row may be fully masked."""
    pad = torch.tensor([[False, False, True, True]])      # 2 pads, then 2 real
    mask = build_attn_mask(q_len=4, offset=0, device="cpu", padding_mask=pad)
    assert mask is not None, "padding must force an explicit mask"
    assert mask.shape == (1, 1, 4, 4), mask.shape

    # Real query rows must not attend to padded columns
    for i in (2, 3):
        for j in (0, 1):
            assert not bool(mask[0, 0, i, j]), f"row {i} attends to pad col {j}"
    # Causality still holds
    assert not bool(mask[0, 0, 2, 3]), "row 2 must not see the future"
    # No fully-masked row (otherwise softmax -> NaN)
    assert bool(mask.any(dim=-1).all()), "a query row is fully masked"

    # Decode step with padding must also produce a mask, not None
    pad2 = torch.tensor([[False, True, True]])
    m2 = build_attn_mask(q_len=1, offset=2, device="cpu", padding_mask=pad2)
    assert m2 is not None and m2.shape == (1, 1, 1, 3)
    assert not bool(m2[0, 0, 0, 0]) and bool(m2[0, 0, 0, 2])

    try:
        build_attn_mask(q_len=4, offset=0, device="cpu",
                        padding_mask=torch.ones(1, 9, dtype=torch.bool))
        raise AssertionError("mismatched padding_mask length should raise")
    except ValueError:
        pass
    ok("padding mask excludes pads, preserves causality, avoids NaN rows")


def test_batched_no_nan():
    """Heavy left padding is where NaN would surface if it were going to."""
    model, cfg = build_model()
    torch.manual_seed(7)
    prompts = [torch.randint(0, cfg.vocab_size, (n,)).tolist() for n in (1, 40)]
    out = model.generate_batch(prompts, max_new_tokens=8, temperature=0.0,
                               eos_token_id=None)
    for row in out:
        assert len(row) == 8
        assert all(0 <= t < cfg.vocab_size for t in row), "invalid token id (NaN logits?)"
    ok("no NaN under extreme padding imbalance (1 vs 40 tokens)")


# ------------------------------------------------------------------ #
# 11. Timing (informational only — never assert on wall clock)
# ------------------------------------------------------------------ #
def report_timing():
    model, cfg = build_model()
    prompt = torch.randint(0, cfg.vocab_size, (1, 256))
    kw = dict(max_new_tokens=32, temperature=0.0, eos_token_id=None)

    t0 = time.time()
    model.generate(prompt, use_cache=False, **kw)
    slow = time.time() - t0

    t0 = time.time()
    model.generate(prompt, use_cache=True, **kw)
    fast = time.time() - t0

    print(f"\n  timing @256 ctx (tiny config): "
          f"uncached {32/slow:.1f} tok/s, cached {32/fast:.1f} tok/s "
          f"({slow/fast:.1f}x)")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 62)
    print("PyCraft-1 KV cache correctness")
    print("=" * 62)

    for fn in (
        test_checkpoint_keys,
        test_mask,
        test_rope_offset,
        test_is_causal_guard,
        test_logit_equivalence,
        test_chunked_prefill,
        test_greedy_equivalence,
        test_seeded_equivalence,
        test_training_contract,
        test_generate_restores_mode,
        test_cache_reuse,
        test_eos_stops,
        test_context_limit,
        test_padding_mask,
        test_batched_equivalence,
        test_batched_per_row_eos,
        test_batched_no_nan,
    ):
        fn()

    report_timing()

    print("\n" + "=" * 62)
    print(f"{_passed} passed, {_skipped} skipped")
    print("=" * 62)
