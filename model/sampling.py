# model/sampling.py
#
# Token sampling for PyCraft-1 generation.
#
# Kept in its own module so the cached and uncached decode paths provably
# share one RNG consumption pattern — that is what makes the cached-vs-
# uncached equivalence test meaningful. If the two paths drew randomness
# differently, an equivalence failure could not distinguish a cache bug from
# a sampling bug.
#
# Filter order is deliberate and matches HuggingFace:
#
#   repetition penalty (on raw logits)
#     -> temperature
#     -> top-k
#     -> top-p
#     -> softmax
#     -> multinomial
#
# Applying the penalty after temperature would change its effective strength;
# applying top-p before top-k would change the candidate set it sees.

import torch


# ------------------------------------------------------------------ #
# Individual filters
# ------------------------------------------------------------------ #
def apply_repetition_penalty(
    logits: torch.Tensor,     # (batch, vocab)
    prev_ids: torch.Tensor,   # (batch, n_prev)
    penalty: float,
) -> torch.Tensor:
    """
    Discourage tokens that have already appeared (CTRL / HuggingFace formula).

    Positive logits are divided by the penalty and negative ones multiplied,
    so both move toward -inf regardless of sign.
    """
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, prev_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(1, prev_ids, score)


def top_k_filter(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    """Keep only the k highest-scoring tokens. k<=0 or k>=vocab disables it."""
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float | None) -> torch.Tensor:
    """
    Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability reaches p. p>=1.0 disables it (and skips a 32k-element sort).
    """
    if p is None or p >= 1.0:
        return logits
    srt, idx = torch.sort(logits, descending=True, dim=-1)
    probs = srt.softmax(dim=-1)
    # Subtracting probs shifts the cumulative sum one position right, which
    # guarantees the top token is always kept even if it alone exceeds p.
    remove = (probs.cumsum(dim=-1) - probs) > p
    srt = srt.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, idx, srt)


# ------------------------------------------------------------------ #
# Combined sampler
# ------------------------------------------------------------------ #
def sample_next_token(
    logits: torch.Tensor,        # (batch, vocab) raw scores
    prev_ids: torch.Tensor,      # (batch, n_prev) tokens so far
    temperature: float = 0.8,    # <= 0.0 selects greedy decoding
    top_k: int | None = 50,
    top_p: float | None = 1.0,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:               # (batch, 1) int64
    """Pick the next token. temperature <= 0.0 means deterministic argmax."""
    logits = apply_repetition_penalty(
        logits.float(), prev_ids, repetition_penalty)

    # Greedy. Handled before the division so temperature=0.0 cannot produce
    # inf/NaN — the old code divided unconditionally and made greedy decoding
    # impossible.
    if temperature is None or temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


# ------------------------------------------------------------------ #
# Quick self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    torch.manual_seed(0)
    V = 100
    logits = torch.randn(1, V)
    prev = torch.tensor([[3, 7, 7]])

    # Greedy is deterministic and matches a plain argmax
    want = int(logits.argmax())
    for temp in (0.0, -1.0, None):
        got = sample_next_token(logits, prev, temperature=temp,
                                repetition_penalty=1.0)
        assert int(got) == want, f"greedy failed at temperature={temp}"
    print("  greedy decoding: OK")

    # top-k restricts the support to exactly k tokens
    filtered = top_k_filter(logits.clone(), 5)
    assert int(torch.isfinite(filtered).sum()) == 5
    assert torch.equal(top_k_filter(logits.clone(), 0), logits), "k=0 disables"
    print("  top_k_filter: OK")

    # top-p keeps at least one token and never more than the full vocab
    for p in (0.01, 0.5, 0.9):
        n = int(torch.isfinite(top_p_filter(logits.clone(), p)).sum())
        assert 1 <= n <= V, f"top_p={p} kept {n} tokens"
    assert torch.equal(top_p_filter(logits.clone(), 1.0), logits), "p=1 disables"
    print("  top_p_filter: OK")

    # Repetition penalty pushes seen tokens down, leaves others untouched
    pen = apply_repetition_penalty(logits.clone(), prev, 2.0)
    for t in (3, 7):
        assert pen[0, t] < logits[0, t], f"token {t} not penalised"
    untouched = [i for i in range(V) if i not in (3, 7)]
    assert torch.equal(pen[0, untouched], logits[0, untouched])
    print("  repetition_penalty: OK")

    # A seeded generator reproduces the same draw
    a = sample_next_token(logits, prev, 0.8, 50, 1.0, 1.0,
                          generator=torch.Generator().manual_seed(42))
    b = sample_next_token(logits, prev, 0.8, 50, 1.0, 1.0,
                          generator=torch.Generator().manual_seed(42))
    assert torch.equal(a, b), "seeded sampling is not reproducible"
    print("  seeded reproducibility: OK")

    print("\nAll sampling tests PASSED.")
