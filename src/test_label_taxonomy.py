"""The three-class taxonomy has three encodings. They must stay in lockstep."""

from pathlib import Path

from src.turntaking_common import (
    CANONICAL_LABELS,
    HF_LABELS,
    LLM_LABELS,
    to_canonical,
)

ROOT = Path(__file__).resolve().parents[1]


def test_every_encoding_has_three_classes():
    for enc in (CANONICAL_LABELS, HF_LABELS, LLM_LABELS):
        assert len(enc) == 3, enc


def test_each_encoding_maps_onto_the_canonical_set():
    for enc in (HF_LABELS, LLM_LABELS):
        assert {to_canonical(x) for x in enc} == set(CANONICAL_LABELS), enc


def test_hf_label_order_matches_the_model_head():
    """HF_LABELS order IS the classifier head order; changing it breaks checkpoints."""
    src = (ROOT / "src/inference_turntaking_hf.py").read_text()
    assert 'LABELS = ["backchannel", "interruption", "none"]' in src
    assert HF_LABELS == ["backchannel", "interruption", "none"]


def test_llm_label_order_matches_the_prompt():
    src = (ROOT / "src/inference_turntaking_llm.py").read_text()
    assert 'LABELS = ["backchannel", "take_floor", "silent"]' in src
    assert LLM_LABELS == ["backchannel", "take_floor", "silent"]


def test_train_head_width_matches_the_taxonomy():
    """The linear head is R^{H x 3}; num_labels must follow LABELS, not a literal."""
    src = (ROOT / "src/train_turntaking_hf.py").read_text()
    assert "config.num_labels = len(LABELS)" in src
