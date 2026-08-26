import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    input_audio: str = "audio.wav"
    working_dir: str = "workdir"
    output_srt: str = "mohra_srt.srt"

    # ------------------------------------------------------------------
    # Hugging Face token (required for pyannote)

    # Create one at: https://huggingface.co/settings/tokens
    # Accept conditions for:
    # https://huggingface.co/pyannote/speaker-diarization-community-1
    # ------------------------------------------------------------------
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))

    # ------------------------------------------------------------------
    # Vocal separation (Demucs)
    # ------------------------------------------------------------------
    do_vocal_separation: bool = True
    demucs_model: str = "htdemucs"

    # ------------------------------------------------------------------
    # Audio cleaning / preprocessing
    # ------------------------------------------------------------------
    target_sr: int = 16000
    do_highpass: bool = True
    highpass_cutoff_hz: float = 80.0
    do_denoise: bool = True
    do_normalize: bool = True
    target_lufs: float = -23.0

    # ------------------------------------------------------------------
    # Whisper (transcription)
    # ------------------------------------------------------------------
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    language: str = "ur"
    beam_size: int = 5
    vad_min_silence_ms: int = 300
    condition_on_previous_text: bool = False

    # ------------------------------------------------------------------
    # Diarization (pyannote)
    # ------------------------------------------------------------------
    # The true speaker count is unknown, so let pyannote estimate it within
    # the configured range. Use --num-speakers only when a count is known.
    # CAUTION: forcing the wrong exact count is one of the most common
    # causes of two people sharing a label or one person's voice getting
    # split across two labels. Check the per-speaker diagnostic printout
    # from Diarizer.run() -- if a label has only 1-2 tiny segments while
    # an exact count is forced, it may not match the true count.
    num_speakers: Optional[int] = None
    min_speakers: int = 2
    max_speakers: int = 10

    # Dedicated pyannote input created from the original, unenhanced audio.
    diarization_target_sample_rate: int = 16000

    # Retained for CLI/config compatibility. Production diarization now always
    # uses a mono 16 kHz PCM-16 WAV prepared from the original input.
    diarization_audio_source: str = "raw"

    # Print every raw diarization segment and every merge/relabel decision.
    diarization_verbose: bool = False

    # Merge diarization segments shorter than this many seconds. 0 disables it.
    # Try 0.3-1.2s if the diagnostic printout shows many short segments.
    diarization_min_segment_duration: float = 0.0

    # ------------------------------------------------------------------
    # VBx clustering hyperparameter overrides (Community-1's internal
    # clustering stage). Diarizer.__init__ prints the pipeline's actual
    # current values via pipeline.parameters(instantiated=True) on every
    # run -- check that printout rather than trusting the numbers in this
    # comment, since pyannote may change its own defaults over time.
    #
    # Leave a field at None to use the pipeline's own default for it.
    #
    # Per Klement et al. 2023 ("Discriminative Training of VBx
    # Diarization"): Fb ("speaker regularization coefficient") directly
    # controls output speaker count -- a HIGHER Fb keeps FEWER speakers.
    # Since our failure mode is under-segmentation (real speakers getting
    # merged into one), LOWERING Fb below its default is the first thing
    # worth testing.
    #
    # Fa ("acoustic scaling factor") mostly affects confidence/sharpness
    # rather than speaker count -- lower priority, more experimental.
    # `threshold` is the separate initial-AHC merge threshold VBx starts
    # from, a different stage of the pipeline than Fa/Fb.
    #
    # Change ONE of these at a time. Changing more than one in the same
    # run makes it impossible to tell which change did what.
    diarization_clustering_threshold: Optional[float] = 0.55
    diarization_clustering_fa: Optional[float] = None
    # The installed Community-1 default (0.8) under-clustered the current
    # recording. 0.5 permits more distinct clusters without forcing a target
    # speaker count; pyannote still estimates the count inside min/max.
    diarization_clustering_fb: Optional[float] = 0.50
    # Native Community-1 segmentation parameter. The benchmark found 0.20
    # reduced tiny fragments for this recording without post-hoc turn deletion.
    diarization_segmentation_min_duration_off: Optional[float] = 0.20

    # Community-1 normally forces every active local embedding into one of
    # the detected speaker clusters. Validate that decision and expose weak
    # evidence as UNKNOWN instead of presenting a nearest-centroid guess as a
    # real identity. Values were chosen conservatively from the current
    # recording's local-embedding score distribution and remain configurable.
    diarization_embedding_validation: bool = True
    diarization_embedding_min_similarity: float = 0.35
    diarization_embedding_min_margin: float = 0.05
    diarization_embedding_max_assignment_gap: float = 0.20

    # Speaker-switch smoothing runs after word alignment and before SRT.
    # Only short, weakly-supported A -> B -> A speaker islands are eligible;
    # strong direct-overlap assignments are protected as genuine short turns.
    enable_label_smoothing: bool = True
    speaker_switch_min_duration: float = 0.6
    speaker_switch_context_words: int = 2
    speaker_switch_context_seconds: float = 1.5
    smoothing_min_majority_ratio: float = 0.66
    speaker_switch_protect_confidence: float = 0.7

    # ------------------------------------------------------------------
    # Speaker identification (optional)
    enable_speaker_identification: bool = True
    embeddings_db_path: str = "embeddings.json"
    identity_similarity_threshold: float = 0.49

    # ------------------------------------------------------------------
    # SRT block chunking
    # ------------------------------------------------------------------
    max_line_duration: float = 6.0
    max_word_gap: float = 0.8
