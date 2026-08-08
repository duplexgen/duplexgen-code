# Data licenses

The **code** in this repository is licensed under Apache-2.0 (see
[`../LICENSE`](../LICENSE)). The **data** released alongside it
(`DuplexGen/duplexgen-corpus`, `DuplexGen/duplexgen-spoken` on the
HuggingFace Hub) is **not** covered by that license: each scenario's
DuplexGen-generated dialogues and slot annotations inherit the license of
the upstream source dataset that scenario's text dialogues were seeded
from.

| Scenario | Code | Source dataset | License |
|---|---|---|---|
| Socratic teaching | TEA | SocraticLM | Apache-2.0 |
| Planning | PLN | MultiWOZ | MIT |
| Interview | INT | Anthropic Interviewer | CC-BY-4.0 |
| Negotiation | NEG | CraigslistBargain | MIT |
| Persuasion | PER | DailyPersuasion | Apache-2.0 |
| Social chat | SOC | SODA | CC-BY-4.0 |

Every record in `duplexgen-corpus` (both the `dialogues/<CODE>/` and
`annotations/<CODE>/` splits) carries a `"license"` field matching the table
above, so the applicable license travels with the record.

## Attribution: Persuasion (PER) split

The Persuasion scenario (`PER`) is seeded from **DailyPersuasion**
(PersuGPT), released under **Apache-2.0**. Please retain attribution to
DailyPersuasion when redistributing the `PER` split.

## Attribution: bundled LibriSpeech audio samples

For convenience, the Stage-5 TTS script ships two small bundles of
**LibriSpeech `train-clean-100`** audio:

- a **tiny 12-speaker sample** (one utterance per speaker) under
  [`../tts_render/librispeech_samples/`](../tts_render/librispeech_samples/) so
  `convert_spoken.py` runs out of the box without downloading the full corpus;
- the fixed assistant voice prompt,
  [`../tts_render/prompt_wavs/assistant_en.wav`](../tts_render/prompt_wavs/assistant_en.wav),
  from a single held-out `train-clean-100` speaker not included in the
  12-speaker pool above (see
  [`../tts_render/prompt_wavs/PROVENANCE.md`](../tts_render/prompt_wavs/PROVENANCE.md)).

Both are part of **LibriSpeech** (Panayotov et al., 2015), released under
**CC BY 4.0**; please retain attribution to LibriSpeech when redistributing
them. For paper-scale rendering, download the full corpus from
<https://www.openslr.org/12>.

## What is (and is not) redistributed

Only **DuplexGen-generated dialogues** (spoken-style conversions and
dialogues with generated turn-taking behavior produced by our pipeline, per
[`../README.md`](../README.md)) and **our own human slot-level preference
annotations** are released. No raw third-party source *text* (e.g. the
original MultiWOZ, CraigslistBargain, SocraticLM, DailyPersuasion, SODA, or
Anthropic Interviewer records) is redistributed in this repository or in the
released HuggingFace datasets. The **one** third-party media exception is the
bundled CC-BY-4.0 LibriSpeech audio described above, bundled purely so the
TTS demo runs without an external download. Users who need the raw
upstream source data, for example to reproduce Stage 1 spoken-style
conversion from scratch, must obtain it directly from the original sources
under their respective licenses.
