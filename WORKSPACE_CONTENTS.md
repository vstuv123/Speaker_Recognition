# Workspace contents

Generated from `D:\WHishper\diarization_pipeline` on 2026-08-17.

> **Status: historical workspace snapshot.** The file list, artifact sizes,
> subtitle observations, configuration defaults, and CLI example below record
> the workspace as it existed on 2026-08-17. They are intentionally preserved
> as project history and must not be read as the current runtime specification.
> For current behavior and commands, use `README.md`. The current standalone
> evaluation tool is documented in `diarization_evaluation/README.md`.

This document inventories the complete workspace at a useful project level. The local `env/` virtual environment contains 43,000+ third-party/runtime files, so it is recorded as an environment directory rather than expanding every installed package file. Binary audio and Python bytecode are listed as artifacts rather than embedded.

## Project overview

This is a speaker-aware transcription pipeline:

```text
input audio -> vocal separation -> cleaning -> diarization -> transcription -> alignment -> SRT
```

## Directory tree

```text
diarization_pipeline/
|-- README.md
|-- requirements.txt
|-- config.py
|-- main.py
|-- audio.wav
|-- audio - Copy.wav
|-- mohra_srt.srt
|-- alignment/
|   |-- __init__.py
|   |-- aligner.py
|   `-- __pycache__/
|-- audio_processing/
|   |-- __init__.py
|   |-- preprocessing.py
|   |-- vocal_separation.py
|   `-- __pycache__/
|-- diarization/
|   |-- __init__.py
|   |-- diarizer.py
|   `-- __pycache__/
|-- subtitles/
|   |-- __init__.py
|   |-- srt_writer.py
|   `-- __pycache__/
|-- transcription/
|   |-- __init__.py
|   |-- transcriber.py
|   `-- __pycache__/
|-- workdir/
|   |-- cleaned.wav
|   `-- separated/htdemucs/audio/
|       |-- vocals.wav
|       `-- no_vocals.wav
|-- env/
|   |-- pyvenv.cfg
|   |-- Scripts/
|   |-- Lib/site-packages/
|   `-- share/
`-- __pycache__/
```

## Source and configuration files

| File | Purpose |
|---|---|
| `README.md` | Setup, usage, pipeline overview, and folder documentation. |
| `requirements.txt` | Python dependencies. |
| `config.py` | Central configuration for paths, preprocessing, Whisper, diarization, and subtitle chunking. |
| `main.py` | CLI and orchestration of the complete six-stage pipeline. |
| `alignment/aligner.py` | Assigns diarization speakers to timestamped words and smooths isolated label changes. |
| `audio_processing/preprocessing.py` | Loads/resamples audio, high-pass filters, denoises, normalizes loudness, and writes cleaned audio. |
| `audio_processing/vocal_separation.py` | Runs Demucs two-stem vocal separation, with a CPU fallback after CUDA failure. |
| `diarization/diarizer.py` | Wraps pyannote Community-1, including an in-memory waveform fallback when TorchCodec is unavailable. |
| `transcription/transcriber.py` | Wraps faster-whisper and returns word-level timestamps. |
| `subtitles/srt_writer.py` | Groups words into speaker-aware subtitle blocks and writes SRT output. |
| Package `__init__.py` files | Empty package markers in each source directory. |

## Dependencies

```text
torch
torchaudio
faster-whisper
pyannote.audio
soundfile
numpy
librosa
scipy
pyloudnorm
noisereduce
demucs
```

## Main configuration defaults

| Setting | Default |
|---|---|
| Input | `audio.wav` |
| Working directory | `workdir` |
| Output | `mohra_srt.srt` |
| Vocal separation | Enabled, `htdemucs` |
| Sample rate | 16,000 Hz |
| High-pass filter | Enabled, 80 Hz |
| Denoising | Enabled |
| Loudness normalization | Enabled, -23 LUFS |
| Whisper model | `large-v3` |
| Whisper device / compute | CUDA / float16 |
| Language | Urdu (`ur`) |
| Beam size | 5 |
| Speaker count | Exactly 7 |
| Subtitle max duration | 6 seconds |
| Subtitle max word gap | 0.8 seconds |

The Hugging Face token is read from the `HF_TOKEN` environment variable and is not stored in this workspace.

## Data and generated artifacts

| Path | Kind | Size / details |
|---|---|---|
| `audio.wav` | Input WAV | 63,352,910 bytes |
| `audio - Copy.wav` | Input WAV copy | 63,352,910 bytes |
| `workdir/cleaned.wav` | Generated cleaned audio | Binary WAV |
| `workdir/separated/htdemucs/audio/vocals.wav` | Demucs vocal stem | Binary WAV |
| `workdir/separated/htdemucs/audio/no_vocals.wav` | Demucs non-vocal stem | Binary WAV |
| `mohra_srt.srt` | Generated subtitle output | 39 speaker-labelled blocks; 4,099 bytes |

The subtitle output spans approximately `00:00:06,570` through `00:05:03,870` and uses speaker labels including `SPEAKER_00`, `SPEAKER_01`, `SPEAKER_04`, `SPEAKER_05`, and `SPEAKER_06`. Its Urdu text appears mojibake-encoded in the current file and may need an encoding repair.

## Runtime-only contents

- `env/` is the local Python virtual environment. It contains Python executables, command-line tools, installed packages, native libraries, metadata, licenses, and manual pages.
- `__pycache__/` directories contain generated `.pyc` bytecode for Python 3.12 and Python 3.13.
- These runtime files are reproducible from Python plus `requirements.txt` and are not application source.

## CLI usage

```powershell
python main.py --input audio.wav --output mohra_srt.srt --num-speakers 7
```

Optional flags are `--language`, `--no-vocal-separation`, `--no-denoise`, and `--device`.

## Current-state reconciliation (2026-08-21)

The project evolved after this inventory was generated:

- Production diarization now uses a dedicated mono, 16 kHz, PCM-16
  `workdir/diarization.wav` prepared from the original input. Demucs vocals and
  `workdir/cleaned.wav` remain on the Whisper transcription branch.
- Speaker count is no longer exactly 7 by default. Current automatic mode uses
  `min_speakers=2` and `max_speakers=10`; `--num-speakers N` remains available
  only when an exact count is known.
- Community-1 VBx assignments now pass through configurable embedding-confidence
  validation. Weak or ambiguous local assignments become `UNKNOWN`, which is
  not counted as an estimated speaker.
- Word/speaker alignment now uses temporal overlap with contextual handling for
  gaps and ambiguity, followed by configurable run-based switch smoothing.
- The repository now contains `tests/` and `diarization_evaluation/`, along with
  additional generated audio, SRT, JSON, CSV, and plot artifacts not present in
  the dated tree above.
- The current verified unit baseline is 39 passed, 0 failed, and 0 skipped
  (2026-08-21).

These notes reconcile the snapshot with the current implementation without
rewriting the historical inventory or its original artifact observations.
