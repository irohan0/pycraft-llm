# model/pycraft_model.py
# PyCraft-1: full autoregressive language model.
#
# Architecture summary:
#   Token embedding
#   → N × TransformerBlock (RMSNorm + GQA/QK-Norm/RoPE + SwiGLU)
#   → Final RMSNorm
#   → Linear output projection (vocab logits)
#
# Training objective: causal language modelling (next-token prediction)
# + Fill-in-the-Middle (FIM) on 50% of batches (handled in data pipeline).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import PyCraftConfig, get_config_120m, get_config_tiny
from model.attention import RMSNorm
from model.transformer import TransformerBlock
from model.kv_cache import KVCache, build_attn_mask
from model.sampling import sample_next_token


class PyCraftModel(nn.Module):
    def __init__(self, config: PyCraftConfig):
        super().__init__()
        self.config = config

        # Token embedding table
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final layer norm before output projection
        self.norm_final = RMSNorm(config.d_model)

        # Output projection: d_model → vocab_size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (optional): share embedding and lm_head weights
        # Saves ~16M params but slightly reduces flexibility.
        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        """
        GPT-2 style initialisation:
        - Embeddings: N(0, 0.02)
        - Linear layers: N(0, 0.02)
        - Residual projections scaled by 1/sqrt(2 * n_layers)
          to keep activations stable as depth increases.
        """
        std = 0.02
        residual_scale = std / math.sqrt(2 * self.config.n_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Scale down output projections (wo and down_proj)
                # which feed directly into residual connections
                if "wo" in name or "down_proj" in name:
                    nn.init.normal_(module.weight, mean=0.0,
                                    std=residual_scale)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,                  # (batch, seq_len)
        # (batch, seq_len) for training
        targets: torch.Tensor | None = None,
        past_key_values=None,                     # KVCache for incremental decoding
        use_cache: bool | None = None,            # None = auto (on iff a cache given)
        num_logits_to_keep: int = 0,              # 0 = all positions
        ignore_index: int = -100,                 # label value excluded from loss
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: token indices, shape (batch, seq_len)
            targets:   next-token targets for loss computation.
                       If None, returns logits only (inference mode).

        Returns:
            (logits, loss)
            logits: (batch, seq_len, vocab_size)
            loss:   scalar cross-entropy loss, or None if targets not given
        """
        batch, seq_len = input_ids.shape

        using_cache = ((past_key_values is not None)
                       if use_cache is None else use_cache)
        if using_cache and past_key_values is None:
            raise ValueError(
                "use_cache=True requires past_key_values; build one with "
                "KVCache.from_model(model, batch_size=...)"
            )

        offset = past_key_values.seq_len if using_cache else 0
        if offset + seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence {offset} + {seq_len} exceeds max_seq_len="
                f"{self.config.max_seq_len} (the RoPE tables end there)"
            )

        # 1. Embed tokens
        x = self.token_embedding(input_ids)   # (batch, seq_len, d_model)

        # 2. Build the causal mask ONCE and share it across all blocks.
        #    Returns None on the two fast paths (offset 0, or single-token
        #    decode) where SDPA needs no explicit mask.
        attn_mask = build_attn_mask(seq_len, offset, x.device)

        # 3. Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask,
                      kv_cache=past_key_values if using_cache else None,
                      position_offset=offset)

        # 4. Advance the cache exactly once, after every layer has written.
        if using_cache:
            past_key_values.advance(seq_len)

        # 5. Final norm
        x = self.norm_final(x)

        # 6. Trim before the output projection when only the last positions
        #    are wanted. lm_head is 16.4M of 55.3M params, so prefilling a
        #    200-token prompt otherwise computes 200x32000 logits for one row.
        if num_logits_to_keep > 0:
            if targets is not None:
                raise ValueError(
                    "num_logits_to_keep cannot be combined with targets"
                )
            x = x[:, -num_logits_to_keep:, :]

        # 7. Project to vocabulary logits
        logits = self.lm_head(x)              # (batch, seq_len, vocab_size)

        # 8. Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy:
            # logits:  (batch * seq_len, vocab_size)
            # targets: (batch * seq_len,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )

        return logits, loss

    def _prepare_prompt(self, input_ids: torch.Tensor, max_new_tokens: int):
        """
        Clamp the prompt to the context window and work out how many tokens
        may still be generated. Deterministic and idempotent, so stream() and
        generate() can both call it and agree.
        """
        batch, prompt_len = input_ids.shape
        if batch != 1:
            raise NotImplementedError(
                "batched generation needs left padding plus a key-padding "
                "mask; see build_attn_mask() in model/kv_cache.py"
            )
        max_len = self.config.max_seq_len
        if prompt_len >= max_len:
            # Truncate once, up front — not once per step.
            input_ids = input_ids[:, -(max_len - 1):]
            prompt_len = input_ids.shape[1]
        return input_ids, min(max_new_tokens, max_len - prompt_len)

    @torch.no_grad()
    def stream(
        self,
        input_ids: torch.Tensor,   # (1, prompt_len) — single sequence only
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | list[int] | None = 0,   # 0 = <|endoftext|>
        stop_strings: list[str] | None = None,
        tokenizer=None,            # required only when stop_strings is given
        use_cache: bool = True,
        past_key_values=None,      # reuse an existing KVCache (it is reset)
        seed: int | None = None,
    ):
        """
        Yield generated token ids one at a time.

        This is the single decode implementation — generate() is a thin
        wrapper that collects what this yields. Incremental consumers (an SSE
        endpoint, a live terminal) can consume it directly.

        The prompt is processed in one prefill pass, then each new token costs
        a single-position forward instead of re-running the whole sequence.

        The context limit is a HARD STOP, not a sliding window: the cache
        holds post-RoPE keys, so positions cannot be re-based without
        re-rotating every cached key. The uncached path (use_cache=False)
        keeps the old crop behaviour and exists as a reference for tests.
        """
        was_training = self.training   # must not leave eval() set behind
        self.eval()
        try:
            device = input_ids.device
            if stop_strings and tokenizer is None:
                raise ValueError("stop_strings requires tokenizer=")

            input_ids, budget = self._prepare_prompt(input_ids, max_new_tokens)
            prompt_len = input_ids.shape[1]
            max_len = self.config.max_seq_len

            eos_ids = ([] if eos_token_id is None
                       else [eos_token_id] if isinstance(eos_token_id, int)
                       else list(eos_token_id))

            generator = None
            if seed is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            cache = None
            if use_cache:
                cache = past_key_values or KVCache.from_model(
                    self, batch_size=1,
                    max_len=prompt_len + budget,
                    device=device,
                    dtype=self.token_embedding.weight.dtype,
                )
                cache.reset()

            # ---- prefill: one pass over the whole prompt ----
            logits, _ = self(input_ids, past_key_values=cache,
                             use_cache=use_cache, num_logits_to_keep=1)
            next_logits = logits[:, -1, :]          # (1, vocab_size)

            out = input_ids
            new_ids: list[int] = []

            for _ in range(budget):
                next_token = sample_next_token(
                    next_logits, out, temperature, top_k, top_p,
                    repetition_penalty, generator=generator,
                )                                   # (1, 1)
                token = int(next_token.item())
                out = torch.cat([out, next_token], dim=1)
                new_ids.append(token)
                yield token

                if token in eos_ids:
                    return
                if stop_strings:
                    # Decode the whole generated span, not the newest token:
                    # byte-level BPE can render a suffix differently than the
                    # full sequence. A stop string may still land mid-token,
                    # so callers wanting exact truncation should cut the text.
                    text = tokenizer.decode(new_ids, skip_special_tokens=False)
                    if any(stop in text for stop in stop_strings):
                        return
                if len(new_ids) >= budget:
                    return

                step_input = next_token if use_cache else out[:, -max_len:]
                logits, _ = self(step_input, past_key_values=cache,
                                 use_cache=use_cache, num_logits_to_keep=1)
                next_logits = logits[:, -1, :]
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,   # (1, prompt_len) — single sequence only
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        **kwargs,                  # see stream() for the full set
    ) -> torch.Tensor:
        """
        Autoregressive generation with a KV cache.

        Returns (1, prompt_len + n_new) — the prompt plus what was generated,
        so callers can still slice with out[0, len(prompt_ids):].
        """
        prepared, _ = self._prepare_prompt(input_ids, max_new_tokens)
        new_ids = list(self.stream(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, **kwargs))
        if not new_ids:
            return prepared
        tail = torch.tensor([new_ids], dtype=prepared.dtype,
                            device=prepared.device)
        return torch.cat([prepared, tail], dim=1)

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel()
                        for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ------------------------------------------------------------------ #
# Full model self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 50)
    print("PyCraft-1 Full Model Test")
    print("=" * 50)

    # Test with tiny config first (fast)
    print("\n[1] Testing PyCraft-tiny...")
    cfg_tiny = get_config_tiny()
    model_tiny = PyCraftModel(cfg_tiny).to(device)
    counts = model_tiny.param_count()
    print(f"    Params: {counts['total'] / 1e6:.2f}M total, "
          f"{counts['trainable'] / 1e6:.2f}M trainable")

    batch, seq = 2, 128
    ids = torch.randint(0, cfg_tiny.vocab_size, (batch, seq), device=device)
    targets = torch.randint(0, cfg_tiny.vocab_size,
                            (batch, seq), device=device)

    logits, loss = model_tiny(ids, targets)
    print(f"    Logits shape: {tuple(logits.shape)}")
    print(
        f"    Loss: {loss.item():.4f}  (expect ~{math.log(cfg_tiny.vocab_size):.2f} for random init)")

    loss.backward()
    print(f"    Backward pass: OK")

    # Test with full 120M config
    print("\n[2] Testing PyCraft-1 (120M)...")
    cfg = get_config_120m()
    model = PyCraftModel(cfg).to(device)
    counts = model.param_count()
    print(f"    Params: {counts['total'] / 1e6:.2f}M total")

    # Memory check (CUDA only)
    if device == "cuda":
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1e6

    ids_full = torch.randint(0, cfg.vocab_size, (1, 256), device=device)
    tgt_full = torch.randint(0, cfg.vocab_size, (1, 256), device=device)
    logits_full, loss_full = model(ids_full, tgt_full)
    loss_full.backward()

    if device == "cuda":
        mem_after = torch.cuda.memory_allocated() / 1e6
        print(f"    GPU memory used: {mem_after:.1f} MB")
    print(f"    Loss: {loss_full.item():.4f}")
    print(f"    Logits shape: {tuple(logits_full.shape)}")

    print("\n[3] Testing generation...")
    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
    prompt = torch.randint(0, cfg.vocab_size, (1, 10), device=device)
    generated = model.generate(
        prompt, max_new_tokens=20, temperature=1.0, top_k=50)
    print(
        f"    Prompt len: {prompt.shape[1]}, Generated len: {generated.shape[1]}")

    print("\n" + "=" * 50)
    print("All tests PASSED. PyCraft-1 architecture is ready.")
    print("=" * 50)
