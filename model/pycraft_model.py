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


class PyCraftModel(nn.Module):
    def __init__(self, config: PyCraftConfig):
        super().__init__()
        self.config = config

        # Token embedding table
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
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
        # 1. Embed tokens
        x = self.token_embedding(input_ids)   # (batch, seq_len, d_model)

        # 2. Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final norm
        x = self.norm_final(x)

        # 4. Project to vocabulary logits
        logits = self.lm_head(x)              # (batch, seq_len, vocab_size)

        # 5. Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy:
            # logits:  (batch * seq_len, vocab_size)
            # targets: (batch * seq_len,)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,   # -1 = padding token (masked from loss)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,   # (1, prompt_len) — single sequence only
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Simple greedy / top-k generation for testing.
        Not for production — use a proper sampler later.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            context = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(context)

            # Take logits at last position
            next_logits = logits[:, -1, :] / temperature   # (1, vocab_size)

            # Top-k filtering
            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, top_k)
                threshold = top_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float('-inf'))

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (1, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

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

    # Memory check
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / 1e6

    ids_full = torch.randint(0, cfg.vocab_size, (1, 256), device=device)
    tgt_full = torch.randint(0, cfg.vocab_size, (1, 256), device=device)
    logits_full, loss_full = model(ids_full, tgt_full)
    loss_full.backward()

    mem_after = torch.cuda.memory_allocated() / 1e6
    print(f"    GPU memory used: {mem_after:.1f} MB")
    print(f"    Loss: {loss_full.item():.4f}")
    print(f"    Logits shape: {tuple(logits_full.shape)}")

    print("\n[3] Testing generation...")
    model.eval()
    torch.cuda.empty_cache()
    prompt = torch.randint(0, cfg.vocab_size, (1, 10), device=device)
    generated = model.generate(
        prompt, max_new_tokens=20, temperature=1.0, top_k=50)
    print(
        f"    Prompt len: {prompt.shape[1]}, Generated len: {generated.shape[1]}")

    print("\n" + "=" * 50)
    print("All tests PASSED. PyCraft-1 architecture is ready.")
    print("=" * 50)
