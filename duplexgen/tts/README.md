# TTS rendering (Stage 5)

Stage 5 renders the generated dialogues to audio with **Chatterbox** TTS. Of
the backends explored during development (ChatTTS, CosyVoice3, Qwen3-TTS,
Chatterbox), only Chatterbox is included in this release.

## What lives where

**`chatterbox/`** (in this directory) is the Chatterbox TTS engine,
a vendored and trimmed copy of
[resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) (MIT,
© Resemble AI). It is the importable `chatterbox` Python package; install it
with `pip install -e duplexgen/tts/chatterbox`. The Stage 5 scripts import from
it, e.g. `from chatterbox.tts_turbo import ChatterboxTurboTTS`. See
[`chatterbox/README.md`](chatterbox/README.md) for what was kept or omitted.

**`tts_render/`** at the repository top level (*not* this directory) holds the
Stage 5 scripts and their assets:

| Path | Purpose |
|---|---|
| `convert_spoken.py` | Render each dialogue into `--num_variants` two-channel variants. |
| `librispeech_samples/` | 12-speaker LibriSpeech sample so rendering runs without the full corpus. |
| `prompt_wavs/` | Fixed assistant speaker prompt (`assistant_en`), with provenance. |

## Running

See [`docs/stage5-tts.md`](../../docs/stage5-tts.md) for the exact commands. In
short, from the repository root:

```bash
pip install -e duplexgen/tts/chatterbox
python tts_render/convert_spoken.py \
  --prompt_dir tts_render/prompt_wavs \
  --input_glob 'outputs/generated_dialogues_with_hf_swbd_plus_backchannels/**/*.json' \
  --save_dir outputs/audios_chatterbox \
  --num_variants 10
```
