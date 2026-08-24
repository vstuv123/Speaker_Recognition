# Speaker-Aware Transcription Pipeline

Turns a raw audio or movie track into a word-timed, speaker-labelled `.srt`
file. The project combines Demucs, faster-whisper, and Pyannote Community-1;
speaker labels are anonymous per run rather than actor identities.

```text
original input
|-- prepare mono 16 kHz PCM-16 WAV -> Pyannote Community-1 -> diarization
`-- Demucs vocals -> cleaning -> faster-whisper -> word timestamps
                                      |
diarization + word timestamps -> overlap alignment -> smoothing -> SRT
```

## Folder structure

```
diarization_pipeline/
|-- config.py                        # runtime defaults
|-- main.py                          # CLI and pipeline orchestration
|-- requirements.txt
|-- README.md
|-- WORKSPACE_CONTENTS.md            # dated historical workspace snapshot
|-- tests/                           # unit tests
|-- diarization_evaluation/          # standalone unsupervised evaluator
|-- audio_processing/
|   |-- vocal_separation.py          # Demucs two-stem separation
|   `-- preprocessing.py             # diarization preparation and Whisper cleaning
|-- diarization/
|   `-- diarizer.py                  # Community-1 plus confidence validation
|-- transcription/
|   `-- transcriber.py               # faster-whisper word timestamps
|-- alignment/
|   `-- aligner.py                   # temporal-overlap alignment and smoothing
|-- subtitles/
|   `-- srt_writer.py                # subtitle grouping and SRT output
`-- workdir/                         # generated intermediate audio
```

## Setup

```bash
pip install -r requirements.txt
```

The `demucs` executable installed by the dependency must be available to the
process running `main.py`.

Set your Hugging Face token (needed for pyannote):

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

You must accept the model conditions here first:
https://huggingface.co/pyannote/speaker-diarization-community-1

## Usage

Edit defaults in `config.py`, or use the CLI. The input should be the original
mixed audio, not a pre-generated vocal stem:

Automatic speaker-count estimation is the default because the exact count is
unknown. Pyannote searches between 2 and 10 speakers. The clustering
regularization is tuned to reduce merged voices, but does not target or force
any particular count:

```bash
python main.py --input audio.wav --output mohra_srt.srt
```

You can narrow that automatic range when appropriate:

```bash
python main.py --input audio.wav --auto-speakers --min-speakers 2 --max-speakers 10
```

When the exact count is known, `--num-speakers` takes precedence over the
range:

```bash
python main.py --input audio.wav --num-speakers 4
```

Useful flags:

- `--no-vocal-separation` — skip Demucs when the input is already dialogue-only.
- `--no-denoise` — skip spectral noise reduction; high-pass filtering and
  loudness normalization remain enabled.
- `--no-smoothing` — disable post-alignment speaker-switch smoothing.
- `--verbose-diarization` — print raw segments and merge/relabel decisions.
- `--min-segment-duration N` — enable the optional short-segment merge heuristic;
  its default is `0`, so it is normally disabled.
- `--device cpu` — request CPU execution. The current default Whisper compute
  type remains `float16`, so CPU-only operation may require changing
  `whisper_compute_type` in `config.py`; it is not covered by the unit suite.

`--diarize-source` is retained only for CLI compatibility. Its value is ignored:
production diarization always uses `workdir/diarization.wav`, prepared from the
original input.

## Processing stages and outputs

1. `prepare_diarization_audio` creates `workdir/diarization.wav` from the
   original input using channel averaging when needed, 16 kHz sampling, and
   PCM-16 WAV encoding. It does not denoise, normalize, filter, or run Demucs.
2. Demucs creates `workdir/separated/htdemucs/<track>/vocals.wav` for the
   transcription branch. CUDA failure is retried on CPU.
3. `clean_audio` resamples the transcription branch, high-pass filters it,
   optionally denoises it, normalizes loudness, and writes `workdir/cleaned.wav`.
4. Community-1 diarizes the prepared original audio. Automatic mode uses a
   2–10 speaker range; an explicit `--num-speakers N` overrides that range.
5. faster-whisper transcribes `workdir/cleaned.wav` with word timestamps.
6. Each word is assigned by maximum meaningful temporal overlap with speaker
   segments. Context resolves gaps/ties, but an explicit embedding-based
   `UNKNOWN` is preserved.
7. Configurable smoothing corrects only short, weakly supported A-B-A speaker
   islands before SRT generation.
8. The SRT writer starts a new block on speaker changes, long blocks, or large
   word gaps.

### Embedding confidence validation

Community-1's VBx clustering assigns every active local speaker embedding to
a cluster. Confidence validation is enabled by default so weak or near-tied
assignments are emitted as `UNKNOWN` rather than being presented as a real
speaker. `UNKNOWN` is not included in the estimated speaker count.

The defaults are configurable in `config.py` or on the command line:

```bash
python main.py --embedding-min-similarity 0.35 --embedding-min-margin 0.05 --embedding-max-assignment-gap 0.20
```

- Increase minimum similarity/margin to reject more questionable labels.
- Decrease them if too much valid dialogue becomes `UNKNOWN`.
- Use `--no-embedding-validation` to compare against unvalidated Community-1
  output.

These checks reduce unsupported speaker assignments; they do not identify
actors and do not guarantee diarization correctness.

## Current defaults

| Area | Default |
|---|---|
| Whisper | `large-v3`, Urdu (`ur`), beam size 5, word timestamps |
| Devices | CUDA, Whisper `float16` |
| Speaker count | Automatic, minimum 2 and maximum 10 |
| VBx overrides | threshold `0.55`, `Fa` unchanged, `Fb` `0.50` |
| Segmentation | `min_duration_off=0.20` |
| Embedding validation | Enabled (`0.35` similarity, `0.05` margin, `0.20` assignment gap) |
| Label smoothing | Enabled; minimum suspect duration `0.6` seconds |
| SRT chunking | 6-second maximum block duration, 0.8-second maximum word gap |

## Tests

Run the unit suite with:

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -v
```

Current verified baseline (2026-08-21): **39 passed, 0 failed, 0 skipped**. The suite
covers audio preparation, speaker-count configuration, embedding-confidence
validation, overlap alignment, switch smoothing, and evaluator metrics. It is
not an end-to-end quality guarantee for arbitrary audio.

## Notes

- Each stage is a separate module/class, so you can test, swap, or tune any one
  of them (e.g. try a different Demucs model, or a different diarization
  pipeline) without touching the rest.
- `alignment/aligner.py` includes an optional `smooth_speaker_labels()` step
  to reduce isolated speaker "flicker" — enabled by default in `main.py`.
- The standalone evaluator is unsupervised: its scores measure stability and
  fragmentation, not speaker correctness. See `diarization_evaluation/README.md`.
- The installed TorchCodec library currently emits a load warning. The
  diarizer catches the related read failure and supplies a SoundFile-loaded
  waveform dictionary to Community-1 instead.
