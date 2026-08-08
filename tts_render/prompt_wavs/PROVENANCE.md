# Provenance: `assistant_en.wav`

The fixed assistant voice-cloning prompt used for every dialogue rendered by
`tts_render/convert_spoken.py`.

| Field | Value |
|---|---|
| Corpus | LibriSpeech (Panayotov et al., 2015) |
| Subset | `train-clean-100` |
| Speaker ID | 458 (Scott Splavec, M) |
| Chapter ID | 126290 |
| Utterance ID | `458-126290-0003` |
| Path in corpus | `train-clean-100/458/126290/458-126290-0003.flac` |
| Duration | 9.51 s |
| Format | 16 kHz mono WAV (LibriSpeech native rate, FLAC transcoded to WAV) |
| License | CC BY 4.0 |

Transcript is in `assistant_en.txt`, verbatim from `458-126290.trans.txt`.

Speaker 458 is deliberately **held out** of the twelve speakers in
`../librispeech_samples/`, which supply the per-variant *user* voices, so the
assistant and a user variant can never share a voice. See
`ASSISTANT_SPEAKER_ID` in `convert_spoken.py`.

LibriSpeech is released under CC BY 4.0; please retain this attribution when
redistributing. Full corpus: <https://www.openslr.org/12>
