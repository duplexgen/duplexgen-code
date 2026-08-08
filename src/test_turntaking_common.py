from src.turntaking_common import _renorm3, human_probs_to_train_dist


def test_renorm3_normalizes_to_one():
    assert _renorm3(1.0, 1.0, 2.0) == [0.25, 0.25, 0.5]


def test_renorm3_returns_all_silence_on_nonpositive_sum():
    assert _renorm3(0.0, 0.0, 0.0) == [0.0, 0.0, 1.0]
    assert _renorm3(-1.0, 0.0, 0.0) == [0.0, 0.0, 1.0]


def test_human_probs_map_onto_training_label_order():
    # human vocabulary -> training order [backchannel, interruption, none]
    out = human_probs_to_train_dist({"backchannel": 0.2, "take_floor": 0.3, "silent": 0.5})
    assert out == [0.2, 0.3, 0.5]


def test_human_probs_tolerate_missing_and_none_values():
    assert human_probs_to_train_dist({}) == [0.0, 0.0, 1.0]
    assert human_probs_to_train_dist({"backchannel": None, "silent": 1.0}) == [0.0, 0.0, 1.0]


def test_human_probs_tolerate_non_dict_input():
    assert human_probs_to_train_dist(None) == [0.0, 0.0, 1.0]
