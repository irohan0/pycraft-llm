# pycraft/engine.py
#
# The single place PyCraft-1 weights are loaded and text is generated.
# The CLI (pycraft/cli.py) and the REST API (pycraft/server.py) are both
# thin wrappers over this class — there is no second decode implementation.
#
# Everything here is CPU-first. PyCraft-1 is 55M parameters, so CUDA is used
# when present but never required.

import os
import re
from pathlib import Path
from typing import Iterator

import torch

from model.config import get_config_120m
from model.pycraft_model import PyCraftModel
from tokenizer.tokenizer_utils import PyCraftTokenizer

# Repo-relative defaults. Both are overridable, and both fall back to a
# HuggingFace download when absent (see _resolve_weights).
DEFAULT_CHECKPOINT = Path("checkpoints/sft_stage1")
DEFAULT_TOKENIZER = Path("tokenizer/vocab/tokenizer.json")
HF_REPO = "imshadow0/pycraft-1"

# The SFT data (Magicoder) was full of markdown code fences, so the model
# reproduces them. Strip them rather than pretend they are code.
_FENCE = re.compile(r"^\s*```[\w]*\s*\n?|```\s*$", re.MULTILINE)


class PyCraft:
    """
    Loaded PyCraft-1 model ready for generation.

        pc = PyCraft()
        print(pc.generate("def is_palindrome(s):"))
        print(pc.fill_in_middle("def add(a, b):\\n    ", "\\n\\nprint(add(1,2))"))

    Generation uses a KV cache, so cost per token is flat rather than growing
    with context length.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        device: str | None = None,
        quantize: bool = False,
        threads: int | None = None,
    ):
        if threads:
            torch.set_num_threads(threads)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        tok_path = Path(tokenizer_path or DEFAULT_TOKENIZER)
        if not tok_path.exists():
            tok_path = _hf_download("tokenizer/tokenizer.json")
        self.tokenizer = PyCraftTokenizer(tok_path)

        cfg = get_config_120m()
        cfg.vocab_size = self.tokenizer.vocab_size
        cfg.dropout = 0.0
        cfg.fim_prefix_id = self.tokenizer.prefix_id
        cfg.fim_suffix_id = self.tokenizer.suffix_id
        cfg.fim_middle_id = self.tokenizer.middle_id
        self.config = cfg

        weights_file = _resolve_weights(checkpoint)
        from safetensors.torch import load_file
        self.model = PyCraftModel(cfg).to(self.device)
        self.model.load_state_dict(load_file(str(weights_file),
                                             device=self.device))
        self.model.eval()

        self.quantized = False
        if quantize:
            if self.device != "cpu":
                raise ValueError(
                    "dynamic int8 quantization is a CPU path; "
                    "use quantize=False on CUDA"
                )
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.quantized = True

        self.checkpoint = weights_file

    # -------------------------------------------------------------- #
    # Core generation
    # -------------------------------------------------------------- #
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.2,
        top_k: int = 20,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> Iterator[str]:
        """
        Yield text deltas as they are generated.

        Decoding is done over the whole accumulated token list each step and
        the new suffix is yielded, rather than decoding tokens individually —
        byte-level BPE can split a multi-byte character across two tokens, and
        per-token decoding would emit replacement characters at the seam.
        """
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        acc: list[int] = []
        emitted = ""
        for token in self.model.stream(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=[self.tokenizer.eot_id, self.tokenizer.pad_id],
            stop_strings=stop,
            tokenizer=self.tokenizer,
            seed=seed,
        ):
            acc.append(token)
            text = self.tokenizer.decode(acc, skip_special_tokens=True)

            # Truncate exactly at a stop string. The model-level check stops
            # the loop, but the token carrying the stop may also carry text
            # before it that the caller still wants.
            if stop:
                cut = _earliest_stop(text, stop)
                if cut is not None:
                    if cut > len(emitted):
                        yield text[len(emitted):cut]
                    return

            if len(text) > len(emitted):
                yield text[len(emitted):]
                emitted = text

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion and return it as one string."""
        return "".join(self.stream(prompt, **kwargs))

    # -------------------------------------------------------------- #
    # Fill in the Middle
    # -------------------------------------------------------------- #
    def fill_in_middle(
        self,
        prefix: str,
        suffix: str,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> str:
        """
        Infill the gap between `prefix` and `suffix`.

        Half of PyCraft-1's pretraining used the Fill-in-the-Middle objective
        in PSM order, so this is a first-class capability rather than a prompt
        trick. The token layout matches apply_fim() in data/fim_utils.py:

            <fim_prefix> prefix <fim_suffix> suffix <fim_middle> -> middle
        """
        tok = self.tokenizer
        ids = (
            [tok.prefix_id] + tok.encode(prefix)
            + [tok.suffix_id] + tok.encode(suffix)
            + [tok.middle_id]
        )
        max_prompt = self.config.max_seq_len - max_new_tokens - 1
        if len(ids) > max_prompt:
            raise ValueError(
                f"FIM prompt is {len(ids)} tokens but only {max_prompt} fit "
                f"alongside max_new_tokens={max_new_tokens} in a "
                f"{self.config.max_seq_len}-token context"
            )

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        kwargs.setdefault("temperature", 0.2)
        kwargs.setdefault("top_k", 20)
        kwargs.setdefault("repetition_penalty", 1.1)
        # The model does not reliably emit a terminator after the middle span,
        # so also stop at a markdown fence — that is where SFT habits take
        # over and it starts restating the whole function.
        out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=[tok.eot_id, tok.middle_id, tok.pad_id],
            stop_strings=["```"],
            tokenizer=tok,
            **kwargs,
        )
        text = tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True)
        return strip_fences(text.split("```")[0])

    # -------------------------------------------------------------- #
    # Convenience
    # -------------------------------------------------------------- #
    def complete_code(self, prompt: str, **kwargs) -> str:
        """Generate, then strip markdown fences the SFT data taught it."""
        kwargs.setdefault("stop", ["\n# Task:", "\nif __name__"])
        return strip_fences(self.generate(prompt, **kwargs)).strip("\n")

    @property
    def n_params(self) -> int:
        cfg = self.config
        # Recomputed from config: after dynamic quantization the packed
        # int8 modules no longer expose parameters via .parameters().
        return cfg.param_count_approx

    def info(self) -> dict:
        return {
            "model": "PyCraft-1",
            "parameters": "55.3M",
            "context_window": self.config.max_seq_len,
            "vocab_size": self.config.vocab_size,
            "device": self.device,
            "quantized": self.quantized,
            "checkpoint": str(self.checkpoint),
            "threads": torch.get_num_threads(),
        }


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def strip_fences(text: str) -> str:
    """Remove markdown code fences from generated output."""
    return _FENCE.sub("", text)


def _earliest_stop(text: str, stops: list[str]) -> int | None:
    """Index of the earliest stop string in `text`, or None."""
    hits = [text.index(s) for s in stops if s in text]
    return min(hits) if hits else None


def _resolve_weights(checkpoint: str | Path | None) -> Path:
    """
    Resolve weights: an explicit path, then the repo checkpoint, then a
    HuggingFace download. Accepts either a directory or a .safetensors file.
    """
    candidate = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    if candidate.is_dir():
        candidate = candidate / "model.safetensors"
    if candidate.exists():
        return candidate
    if checkpoint is not None:
        raise FileNotFoundError(f"No weights at {candidate}")
    return _hf_download("model.safetensors")


def _hf_download(filename: str) -> Path:
    """Fetch a file from the published HuggingFace repo (cached after first use)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"{filename} not found locally and huggingface_hub is not "
            f"installed. Either run from the repo root (where "
            f"{DEFAULT_CHECKPOINT} lives) or `pip install huggingface_hub`."
        ) from exc
    repo = os.environ.get("PYCRAFT_HF_REPO", HF_REPO)
    print(f"  downloading {filename} from {repo} (first run only)...")
    return Path(hf_hub_download(repo_id=repo, filename=filename))
