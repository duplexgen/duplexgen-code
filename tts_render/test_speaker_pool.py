import pathlib

import pytest


def _load_module():
    """Load list_librispeech_speakers without importing convert_spoken.

    convert_spoken.py calls nltk.download() at import time and pulls in
    whisperx/nemo, so it cannot be imported in a unit test. We exec only the
    function under test from source.
    """
    src = pathlib.Path(__file__).with_name("convert_spoken.py").read_text()
    start = src.index("def list_librispeech_speakers")
    end = src.index("def sample_librispeech_prompt")
    ns = {"os": __import__("os"), "glob": __import__("glob").glob}
    exec(compile(src[start:end], "convert_spoken_excerpt", "exec"), ns)
    return ns["list_librispeech_speakers"]


@pytest.fixture
def fake_corpus(tmp_path):
    for spk in (19, 26, 777):
        d = tmp_path / "train-clean-100" / str(spk) / "1"
        d.mkdir(parents=True)
        (d / f"{spk}-1-0001.flac").write_bytes(b"")
    return tmp_path


def test_pool_contains_all_speakers_without_exclusion(fake_corpus):
    fn = _load_module()
    pool = fn(str(fake_corpus), ["train-clean-100"])
    assert set(pool) == {"19", "26", "777"}


def test_excluded_speaker_is_absent(fake_corpus):
    fn = _load_module()
    pool = fn(str(fake_corpus), ["train-clean-100"], exclude={"777"})
    assert set(pool) == {"19", "26"}
    assert "777" not in pool


def test_exclusion_accepts_ints_and_strings(fake_corpus):
    fn = _load_module()
    pool = fn(str(fake_corpus), ["train-clean-100"], exclude={777})
    assert "777" not in pool
