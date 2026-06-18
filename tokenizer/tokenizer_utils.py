# tokenizer/tokenizer_utils.py
#
# Loads the trained BPE tokenizer and provides a clean interface
# for encoding, decoding, and accessing special token IDs.
# Used by the data pipeline and training loop.

from pathlib import Path
from tokenizers import Tokenizer


TOKENIZER_PATH = Path("tokenizer/vocab/tokenizer.json")

# Special token string constants — single source of truth
TOK_EOT = "<|endoftext|>"
TOK_PREFIX = "<|fim_prefix|>"
TOK_SUFFIX = "<|fim_suffix|>"
TOK_MIDDLE = "<|fim_middle|>"
TOK_PAD = "<|pad|>"


class PyCraftTokenizer:
    """
    Thin wrapper around the HuggingFace tokenizers BPE tokenizer.
    Exposes encode/decode and special token IDs used by the data pipeline.
    """

    def __init__(self, path: str | Path = TOKENIZER_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {path}.\n"
                f"Run: python -m tokenizer.train_tokenizer"
            )
        self._tok = Tokenizer.from_file(str(path))

        # Cache special token IDs
        self.eot_id = self._id(TOK_EOT)
        self.prefix_id = self._id(TOK_PREFIX)
        self.suffix_id = self._id(TOK_SUFFIX)
        self.middle_id = self._id(TOK_MIDDLE)
        self.pad_id = self._id(TOK_PAD)

        # Disable truncation/padding at tokenizer level
        # (handled manually in data pipeline)
        self._tok.no_truncation()
        self._tok.no_padding()

    def _id(self, token: str) -> int:
        tok_id = self._tok.token_to_id(token)
        assert tok_id is not None, f"Special token {token!r} not in vocab!"
        return tok_id

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Encode a string to a list of token IDs."""
        return self._tok.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to a string."""
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def __repr__(self) -> str:
        return (
            f"PyCraftTokenizer(vocab_size={self.vocab_size}, "
            f"eot={self.eot_id}, prefix={self.prefix_id}, "
            f"suffix={self.suffix_id}, middle={self.middle_id})"
        )


# ------------------------------------------------------------------ #
# Quick self-test (only runs after tokenizer is trained)
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Loading tokenizer...")
    tok = PyCraftTokenizer()
    print(tok)
    print()

    samples = [
        "def hello_world():\n    print('Hello, world!')\n",
        "import numpy as np\nx = np.zeros((3, 3))\n",
        "# PyCraft-1 tokenizer test\nfor i in range(10):\n    pass\n",
    ]

    for code in samples:
        ids = tok.encode(code)
        decoded = tok.decode(ids)
        print(f"  Original : {repr(code[:50])}")
        print(f"  Tokens   : {len(ids)}")
        print(f"  Decoded  : {repr(decoded[:50])}")
        print()
