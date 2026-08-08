"""Guards against the environment contradiction that made Stage 3 uninstallable.

Stages 1-4 and Stage 5 are separate environments on purpose. These tests fail if
anyone re-merges them or lowers the transformers floor below what
transformers.modeling_layers requires.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELING_LAYERS_MIN = (4, 53)  # transformers.modeling_layers first shipped in 4.53


def _parse(req_path):
    """Return {package: full_specifier} for a pip requirements file."""
    out = {}
    for line in req_path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip().lower()
        out[name] = line
    return out


def _floor(spec):
    m = re.search(r">=\s*(\d+)\.(\d+)", spec)
    assert m, f"no >= floor in {spec!r}"
    return (int(m.group(1)), int(m.group(2)))


def test_stage14_transformers_floor_supports_modeling_layers():
    spec = _parse(ROOT / "requirements.txt")["transformers"]
    assert _floor(spec) >= MODELING_LAYERS_MIN, (
        f"requirements.txt pins {spec}, but src/train_turntaking_hf.py imports "
        "transformers.modeling_layers, which requires >=4.53"
    )


def test_stage14_transformers_has_an_upper_bound():
    spec = _parse(ROOT / "requirements.txt")["transformers"]
    assert "<" in spec, f"transformers needs a major upper bound, got {spec!r}"


def test_stage5_requirements_do_not_reinstall_stage14_packages():
    """The two files must not both pin transformers into one env."""
    stage5 = _parse(ROOT / "requirements-stage5.txt")
    assert "transformers" not in stage5, (
        "requirements-stage5.txt must not pin transformers; the vendored "
        "chatterbox pyproject owns that pin in its own environment"
    )


def test_vendored_chatterbox_still_pins_transformers_exactly():
    """If upstream's pin ever moves, the split may no longer be necessary."""
    text = (ROOT / "duplexgen/tts/chatterbox/pyproject.toml").read_text()
    assert "transformers==4.46.3" in text, (
        "vendored chatterbox pin changed; re-evaluate whether two environments "
        "are still required"
    )


def test_readme_does_not_tell_users_to_install_both_into_one_env():
    readme = (ROOT / "README.md").read_text()
    setup = readme.split("## Setup", 1)[1].split("###", 1)[0]
    assert "requirements-stage5.txt" in setup, "Setup must document the Stage 5 env"
    assert "3.11" in setup, "Setup must state the Python 3.11 requirement"


def test_constraints_cover_every_direct_requirement():
    """Every Stage 1-4 package we ask for must have a pinned, tested version.

    Stage 5 is excluded on purpose: constraints.txt pins numpy==2.0.2,
    torch==2.7.1 and transformers==4.57.6, which contradict the versions the
    vendored Chatterbox pyproject requires for its own environment. Scoping
    constraints.txt to requirements.txt keeps it from re-asserting pins that
    Stage 5 must not use.
    """
    constraints = _parse(ROOT / "constraints.txt")
    direct = set(_parse(ROOT / "requirements.txt"))
    missing = sorted(direct - set(constraints))
    assert not missing, f"constraints.txt is missing pins for: {missing}"


def test_stage5_requirements_bound_numpy_below_2():
    """Under NumPy 2's NEP 50 casting, Chatterbox's norm_loudness() upcasts its
    float32 waveform to float64 and no longer matches the s3tokenizer's float32
    mel filters, so Stage 5 dies with 'expected scalar type Double but found
    Float'. This bound must not be dropped.
    """
    stage5 = _parse(ROOT / "requirements-stage5.txt")
    assert "numpy" in stage5, "requirements-stage5.txt must pin numpy"
    assert "<2" in stage5["numpy"], f"numpy must be bounded below 2, got {stage5['numpy']!r}"


def test_constraints_are_exact_pins():
    for name, spec in _parse(ROOT / "constraints.txt").items():
        assert "==" in spec, f"{name} must be an exact pin in constraints.txt, got {spec!r}"


def test_readme_documents_system_dependencies():
    readme = (ROOT / "README.md").read_text()
    for tool in ("pynini", "OpenFst"):
        assert tool in readme, f"README must document the {tool} system dependency"
