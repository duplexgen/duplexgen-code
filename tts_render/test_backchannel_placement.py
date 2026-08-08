import pathlib

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


def _load_placement():
    """Load generate_delay and place_backchannel without importing convert_spoken.

    convert_spoken.py calls nltk.download() at import time and pulls in
    whisperx/nemo, so it cannot be imported in a unit test. We exec only the
    functions under test from source.
    """
    src = pathlib.Path(__file__).with_name("convert_spoken.py").read_text()
    start = src.index("def generate_delay")
    end = src.index("def aggregate_speech")
    ns = {"np": np, "torch": torch, "F": torch.nn.functional, "TARGET_SR": 24000}
    exec(compile(src[start:end], "convert_spoken_excerpt", "exec"), ns)
    return ns["generate_delay"], ns["place_backchannel"]


def _track(n):
    return torch.zeros(1, n)


def test_delay_is_never_negative_for_a_backchannel_longer_than_its_gap():
    generate_delay, _ = _load_placement()
    assert generate_delay(-5000, mode="backchannel") == 0
    assert generate_delay(5000, mode="backchannel") == 0


def test_backchannel_lands_where_it_is_asked_to():
    _, place_backchannel = _load_placement()
    listener, host = place_backchannel(_track(100), _track(100), torch.ones(1, 10), 30)
    assert listener.size(1) == host.size(1) == 100
    assert listener[0, 30:40].tolist() == [1.0] * 10
    assert listener[0, :30].sum() == 0 and listener[0, 40:].sum() == 0


def test_overrunning_backchannel_grows_both_tracks_instead_of_raising():
    _, place_backchannel = _load_placement()
    listener, host = place_backchannel(_track(100), _track(100), torch.ones(1, 40), 80)
    # 80 + 40 - 100 = 20 samples of overrun; both channels stay the same length so
    # they can still be stacked into one stereo tensor.
    assert listener.size(1) == host.size(1) == 120
    assert listener[0, 80:120].tolist() == [1.0] * 40


def test_backchannel_longer_than_the_host_turn_is_kept_whole():
    _, place_backchannel = _load_placement()
    listener, host = place_backchannel(_track(10), _track(10), torch.ones(1, 50), 0)
    assert listener.size(1) == host.size(1) == 50
    assert listener[0].tolist() == [1.0] * 50


def test_negative_start_is_clamped_rather_than_wrapping_to_the_end():
    _, place_backchannel = _load_placement()
    listener, _ = place_backchannel(_track(100), _track(100), torch.ones(1, 10), -20)
    assert listener[0, :10].tolist() == [1.0] * 10
    assert listener[0, 10:].sum() == 0
