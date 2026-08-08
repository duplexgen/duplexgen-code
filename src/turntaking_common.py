"""Shared turn-taking probability helpers.

Single definition for conversions between the human annotation label
vocabulary (backchannel / take_floor / silent) and the training label order
(backchannel / interruption / none). Previously duplicated verbatim across
train_turntaking_hf.py, inference_turntaking_hf.py, and
inference_turntaking_llm.py.
"""

from typing import Any, Dict, List


def _renorm3(a: float, b: float, c: float) -> List[float]:
    s = a + b + c
    if s <= 0:
        return [0.0, 0.0, 1.0]
    return [a / s, b / s, c / s]


def human_probs_to_train_dist(probabilities: Dict[str, Any]) -> List[float]:
    probs = probabilities if isinstance(probabilities, dict) else {}
    bc = float(probs.get("backchannel", 0.0) or 0.0)
    intr = float(probs.get("take_floor", 0.0) or 0.0)
    none = float(probs.get("silent", 0.0) or 0.0)
    return _renorm3(bc, intr, none)


# --------------------------------------------------------------------------
# Label taxonomy.
#
# The same three classes are spelled three different ways across this codebase.
# None of the spellings can be changed unilaterally:
#
#   HF_LABELS           order IS the classifier head order, baked into every
#                       trained checkpoint (src/inference_turntaking_hf.py:24)
#   LLM_LABELS          the vocabulary the verbalized-probability prompt asks
#                       the model to emit (src/inference_turntaking_llm.py:16)
#
# Use to_canonical() at every boundary rather than comparing strings.
# --------------------------------------------------------------------------

CANONICAL_LABELS = ["backchannel", "floor_taking", "silence"]

HF_LABELS = ["backchannel", "interruption", "none"]
LLM_LABELS = ["backchannel", "take_floor", "silent"]

_TO_CANONICAL = {
    "backchannel": "backchannel",
    "interruption": "floor_taking",
    "take_floor": "floor_taking",
    "floor_taking": "floor_taking",
    "none": "silence",
    "silent": "silence",
    "silence": "silence",
}


def to_canonical(label: str) -> str:
    """Map any of the four spellings onto the canonical label."""
    try:
        return _TO_CANONICAL[label]
    except KeyError:
        raise ValueError(
            f"unknown turn-taking label {label!r}; expected one of " f"{sorted(_TO_CANONICAL)}"
        ) from None
