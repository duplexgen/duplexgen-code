# Chatterbox TTS (vendored)

This directory is a **vendored copy** of
[Chatterbox](https://github.com/resemble-ai/chatterbox) TTS by Resemble AI,
trimmed to just the importable `chatterbox` Python package that DuplexGen
Stage 5 depends on. It is licensed under the **MIT License** (see
[`LICENSE`](LICENSE)); copyright © 2025 Resemble AI.

Only the package source (`src/chatterbox/`) and its `pyproject.toml` are kept
here — the upstream demo assets, example/Gradio apps, and voice samples are
omitted. For the complete project, examples, and history, see the original
repository:

- **Upstream:** <https://github.com/resemble-ai/chatterbox>

## Install

From the DuplexGen repository root:

```bash
pip install -e duplexgen/tts/chatterbox
```

This provides `from chatterbox.tts_turbo import ChatterboxTurboTTS` (and the
other imports used by the top-level `chatterbox/convert_spoken.py` Stage-5
script). See [`docs/stage5-tts.md`](../../../docs/stage5-tts.md) for how Stage 5
is run.
