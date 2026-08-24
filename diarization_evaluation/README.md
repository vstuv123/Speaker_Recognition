# Diarization parameter evaluation

This directory contains a standalone, unsupervised benchmark. It does not
change production configuration and is not the production diarization path.

```powershell
.\env\Scripts\python.exe diarization_evaluation\evaluate.py --audio audio.wav
```

Whisper runs once and its word timestamps are cached. Community-1 is loaded
once and re-instantiated for each tested threshold, native
`segmentation.min_duration_off`, and speaker-count mode. The current evaluator
tests:

- clustering thresholds `0.55`, `0.60`, `0.65`, and `0.70`;
- `segmentation.min_duration_off` values `0.00`, `0.10`, `0.20`, and `0.30`
  at the best heuristic threshold;
- automatic speaker estimation from 2–7 versus an exact-7 comparison at the
  selected threshold/minimum-off combination.

Duplicate configurations are reused, so a normal run produces eight unique
configuration directories. Results, per-run SRTs, diagnostics, and plots are
written below this directory.

Scores are heuristic stability scores, not diarization error rate. Listen to
the candidates before applying any recommendation to production.

Important current-state distinction: production defaults to automatic 2–10
speaker estimation, uses VBx `Fb=0.50`, and applies embedding-confidence
validation. This evaluator's historical experiment grid does not test `Fb` and
does not use the production confidence wrapper. Its exact-7 experiment is a
comparison candidate, not a statement that the recording or production system
has seven speakers.

Generated files include `whisper_words.json`, `results.json`, `results.csv`,
`summary.txt`, plots, and per-configuration `configuration.json`,
`diarization.json`, `aligned_words.json`, `diagnostics.json`, and `output.srt`.
