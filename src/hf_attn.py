"""Attention-backend selection for the Stage 3 HuggingFace predictor.

`flash_attention_2` needs the optional `flash-attn` wheel, which is not in
requirements.txt: it has no universal manylinux wheel and compiles from source
against the installed CUDA toolkit, so it cannot be a hard dependency of a
`pip install -r requirements.txt` setup. The Stage 3 scripts used to hardcode
it, which made a clean install fail at model load with

    ImportError: FlashAttention2 has been toggled on, but it cannot be used ...

Both entry points now default to "auto", which uses FlashAttention-2 when the
package is importable and otherwise falls back to PyTorch SDPA.
"""

import importlib.util

import torch

CHOICES = ["auto", "flash_attention_2", "sdpa", "eager"]


def flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def resolve_attn_implementation(preferred: str = "auto") -> str:
    """Map `preferred` onto a backend that is actually usable here."""
    if preferred != "auto":
        return preferred
    if not torch.cuda.is_available():
        return "eager"
    return "flash_attention_2" if flash_attn_available() else "sdpa"
