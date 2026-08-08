# Bundled LibriSpeech samples

A tiny 12-speaker subset of LibriSpeech `train-clean-100` (one utterance per
speaker, each > 5 s), shipped so `convert_spoken.py` runs out of the box without
downloading the full corpus. Layout mirrors LibriSpeech:

```
librispeech_samples/train-clean-100/<speaker>/<chapter>/<utt>.flac
```

`convert_spoken.py` defaults `--librispeech_root` to this directory. Twelve
speakers is exactly enough for the default `--num_variants 10` (each dialogue
variant uses a distinct user voice).

The assistant's fixed voice prompt (`../prompt_wavs/assistant_en.wav`) comes
from a **different** `train-clean-100` speaker, deliberately held out of these
twelve so the assistant and a user variant can never share a voice.

## Using the full corpus

For paper-scale rendering, download LibriSpeech `train-clean-100` and
`train-clean-360` from <https://www.openslr.org/12> and point the script at your
copy:

```bash
python tts_render/convert_spoken.py \
  --librispeech_root /path/to/LibriSpeech \
  --librispeech_subsets train-clean-100,train-clean-360 \
  ...
```

These files are part of LibriSpeech (Panayotov et al., 2015), released under
CC BY 4.0.
