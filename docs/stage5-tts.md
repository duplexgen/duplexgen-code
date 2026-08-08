# Stage 5: TTS rendering

Renders the Stage 4 result to two-channel audio with Chatterbox.

```bash
.venv-tts/bin/python tts_render/convert_spoken.py \
  --prompt_dir tts_render/prompt_wavs \
  --input_glob 'outputs/generated_dialogues_with_hf_swbd_plus_backchannels/**/*.json' \
  --save_dir outputs/audios_chatterbox \
  --num_variants 10
```

Note the **other interpreter** (`.venv-tts`, see [Setup](../README.md#setup))
and that this is run as a *file path*, not as a `-m` module like Stages 1-4.

Each dialogue is rendered into `--num_variants` two-channel variants: the
assistant uses a fixed cloning prompt (`tts_render/prompt_wavs/assistant_en.wav`,
a held-out LibriSpeech speaker — see `tts_render/prompt_wavs/PROVENANCE.md`) and
each variant's user voice is sampled from a distinct LibriSpeech speaker.
Backchannels are synthesized by Chatterbox through the same voice prompt as the
speaker uttering them, so they match that variant's voice.

> [!IMPORTANT]
> **Chatterbox renders backchannels poorly.** Short interjections — `mhm`,
> `mm-hmm` — come out flat and mispronounced; a zero-shot TTS has almost no
> context to work from on a two-phoneme utterance. **The paper's dialogues used
> the ElevenLabs API for these instead.** That dependency was dropped for 1.0 so
> the pipeline runs without a paid key, and what ships is the self-contained
> fallback. To use another backend, replace the `uttr_type == "backchannel"`
> branch in `main_process` (`tts_render/convert_spoken.py`) — placement, caching
> and the `backchannels/*.wav` outputs take whatever waveform it returns. Keep
> the backchannel voice matched to the speaker uttering it.

## Output layout

`--input_glob` is matched against the Stage 4 output layout, and the trailing
`text_dialogue_<dataset>/<split>/` of each input path is mirrored under
`--save_dir`:

```
outputs/audios_chatterbox/text_dialogue_interviewer/train/work_0000/var00/
├── dialogues/dialogue.wav     # the full two-channel mix
├── utterances/00.wav ...      # one two-channel file per utterance, pre-mix
├── backchannels/05_0.wav ...  # <utterance>_<slot>, the isolated backchannel
└── meta.json                  # per-utterance text, timings, and the voice used
```

`dialogue.wav` is 24 kHz 16-bit stereo with **channel 0 = assistant, channel 1 =
user** — the channels are swapped on the way out so the assistant lands first,
which is the order `personaplex-finetune` expects. `meta.json` records that as
`"speakers": ["assistant", "user"]`, along with `user_prompt_wav` and
`user_prompt_speaker_id` for the variant's LibriSpeech voice.

A variant whose `dialogue.wav` and `meta.json` both exist is skipped, so a
re-run resumes rather than re-synthesizing. Voice casting is seeded per dialogue
from `--seed` and the file name, so a resumed run fills the missing variants
from the same cast as the ones already on disk.

## Cost knobs

Rendering runs at roughly real time, so cap a trial run: `--max_dialogues N`
takes only the first N dialogues, and `--num_variants` scales the per-dialogue
cost linearly. To spread a full render over several GPUs, give each worker the
same command with `--num_shards W --shard_id I`; shards partition the file list,
and since each writes its own dialogue directories they can share one
`--save_dir`. `--input_glob` accepts several globs, which are concatenated and
de-duplicated, and `--exclude_ids_file` skips dialogue stems listed one per
line.

## Dropped inputs

Dialogues carrying LLM artifacts the sanitizer cannot repair — leaked think
blocks, concatenated turns, template placeholders — are skipped with a
`DROP-DIALOGUE` warning naming the reason. Individual backchannels are dropped
with a warning when forced alignment returns no words to anchor them to; the
dialogue still renders, and `utterances_with_bc[i].texts_styled[j].isUttered` in
`meta.json` records which backchannels actually made it into the audio. The run
ends with a summary line counting rendered, skipped, dropped, and failed
variants, and exits non-zero if it rendered nothing at all.

## Voices

By default `--librispeech_root` points at the 12-speaker sample bundled in
`tts_render/librispeech_samples/`, so this runs out of the box. That sample
bounds `--num_variants` at 12; for paper-scale rendering, download the full
[LibriSpeech](https://www.openslr.org/12) corpus and pass `--librispeech_root
/path/to/LibriSpeech --librispeech_subsets train-clean-100,train-clean-360`.
