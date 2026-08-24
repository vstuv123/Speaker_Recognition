"""
Orchestrates the full pipeline:

  vocal separation -> audio cleaning -> diarization -> transcription
  -> word/speaker alignment -> SRT generation

NOTE: diarization runs on a mono 16 kHz PCM-16 copy of the original input,
not the vocal-separated/cleaned version used for transcription.

Usage:
    python main.py
    python main.py --input my_audio.wav --output out.srt --num-speakers 5
"""

import argparse
import os

from config import Config
from audio_processing.vocal_separation import separate_vocals
from audio_processing.preprocessing import clean_audio, prepare_diarization_audio
from diarization.diarizer import Diarizer
from transcription.transcriber import Transcriber
from alignment.aligner import assign_speakers_to_words, smooth_speaker_labels
from subtitles.srt_writer import build_srt_blocks, write_srt


def parse_args(cfg: Config, argv=None):
    parser = argparse.ArgumentParser(description="Speaker-aware transcription -> SRT pipeline")
    parser.add_argument("--input", default=cfg.input_audio, help="Path to input audio file")
    parser.add_argument("--output", default=cfg.output_srt, help="Path to output .srt file")
    parser.add_argument(
        "--num-speakers", type=int, default=cfg.num_speakers,
        help="Use an exact speaker count. When supplied, this takes precedence "
             "over --min-speakers/--max-speakers.",
    )
    parser.add_argument(
        "--auto-speakers", action="store_true",
        help="Use automatic speaker estimation with --min-speakers/--max-speakers "
             "(retained for compatibility; automatic mode is now the default).",
    )
    parser.add_argument("--min-speakers", type=int, default=cfg.min_speakers)
    parser.add_argument("--max-speakers", type=int, default=cfg.max_speakers)
    parser.add_argument("--language", default=cfg.language)
    parser.add_argument("--no-vocal-separation", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--device", default=cfg.whisper_device)
    parser.add_argument(
        "--min-segment-duration", type=float, default=cfg.diarization_min_segment_duration,
        help="Merge diarization segments shorter than this many seconds (0 = off).",
    )
    parser.add_argument(
        "--enable-smoothing", action="store_true", default=cfg.enable_label_smoothing,
        help=argparse.SUPPRESS,  # kept for backward compatibility; smoothing is on by default now
    )
    parser.add_argument(
        "--no-smoothing", action="store_true",
        help="Disable label smoothing (it's on by default -- see aligner.smooth_speaker_labels).",
    )
    parser.add_argument(
        "--diarize-source", choices=["raw", "vocals", "cleaned"],
        default=cfg.diarization_audio_source,
        help="Legacy option retained for compatibility; diarization always uses "
             "a minimally converted copy of the original audio.",
    )
    parser.add_argument(
        "--verbose-diarization", action="store_true",
        help="Print every raw diarization segment and every merge/relabel decision "
             "-- use this to inspect exactly what happened at a specific timestamp.",
    )
    parser.add_argument(
        "--clustering-threshold", type=float, default=cfg.diarization_clustering_threshold,
        help="Override pyannote's VBx clustering.threshold (initial AHC merge "
             "threshold). The pipeline's own current value is printed at startup "
             "-- leave unset to use it as-is.",
    )
    parser.add_argument(
        "--clustering-fa", type=float, default=cfg.diarization_clustering_fa,
        help="Override VBx clustering.Fa (acoustic scaling factor). Mostly affects "
             "confidence/sharpness rather than speaker count. Leave unset to use "
             "the pipeline default.",
    )
    parser.add_argument(
        "--clustering-fb", type=float, default=cfg.diarization_clustering_fb,
        help="Override VBx clustering.Fb (speaker regularization coefficient) -- "
             "a HIGHER Fb keeps FEWER speakers, so LOWER this if real speakers are "
             "getting merged. Leave unset to use the pipeline default.",
    )
    parser.add_argument(
        "--no-embedding-validation", action="store_true",
        help="Disable confidence validation of Community-1 speaker embeddings.",
    )
    parser.add_argument(
        "--embedding-min-similarity", type=float,
        default=cfg.diarization_embedding_min_similarity,
        help="Minimum cosine similarity required for a local embedding to keep "
             "its VBx speaker label (default: %(default)s).",
    )
    parser.add_argument(
        "--embedding-min-margin", type=float,
        default=cfg.diarization_embedding_min_margin,
        help="Minimum best-vs-second centroid margin; lower-confidence ties are "
             "reported as UNKNOWN (default: %(default)s).",
    )
    parser.add_argument(
        "--embedding-max-assignment-gap", type=float,
        default=cfg.diarization_embedding_max_assignment_gap,
        help="Maximum similarity loss allowed when VBx overlap constraints choose "
             "a non-nearest centroid (default: %(default)s).",
    )
    argv_values = list(argv) if argv is not None else None
    args = parser.parse_args(argv_values)
    inspected_argv = argv_values
    if inspected_argv is None:
        import sys
        inspected_argv = sys.argv[1:]
    args.num_speakers_explicit = any(
        value == "--num-speakers" or value.startswith("--num-speakers=")
        for value in inspected_argv
    )
    return args


def apply_speaker_count_args(cfg: Config, args):
    """Apply CLI speaker-count options with explicit exact count precedence."""
    if args.min_speakers < 1:
        raise ValueError("--min-speakers must be at least 1")
    if args.max_speakers < args.min_speakers:
        raise ValueError("--max-speakers must be greater than or equal to --min-speakers")
    if args.num_speakers is not None and args.num_speakers < 1:
        raise ValueError("--num-speakers must be at least 1")

    if args.num_speakers_explicit:
        cfg.num_speakers = args.num_speakers
    elif args.auto_speakers:
        cfg.num_speakers = None
    else:
        cfg.num_speakers = args.num_speakers

    cfg.min_speakers = args.min_speakers
    cfg.max_speakers = args.max_speakers


def run_pipeline(cfg: Config):
    os.makedirs(cfg.working_dir, exist_ok=True)

    # Diarization gets a dedicated minimally converted copy of the original.
    # Transcription continues through the existing Demucs/cleaning branch.
    raw_audio_path = cfg.input_audio
    audio_path = cfg.input_audio
    diarize_audio_path = prepare_diarization_audio(
        raw_audio_path,
        os.path.join(cfg.working_dir, "diarization.wav"),
        target_sample_rate=cfg.diarization_target_sample_rate,
    )

    # 1. Vocal separation - strips background music/effects, leaving dialogue.
    if cfg.do_vocal_separation:
        sep_dir = os.path.join(cfg.working_dir, "separated")
        audio_path = separate_vocals(
            audio_path,
            sep_dir,
            model=cfg.demucs_model,
            device=cfg.whisper_device,
        )

    # 2. Cleaning - resample, high-pass filter, denoise, normalize loudness.
    cleaned_path = os.path.join(cfg.working_dir, "cleaned.wav")
    audio_path = clean_audio(
        audio_path,
        cleaned_path,
        target_sr=cfg.target_sr,
        do_highpass=cfg.do_highpass,
        highpass_cutoff=cfg.highpass_cutoff_hz,
        do_denoise=cfg.do_denoise,
        do_normalize=cfg.do_normalize,
        target_lufs=cfg.target_lufs,
    )

    # 3. Diarization - who spoke when. Always use the prepared original;
    # never Demucs vocals or the aggressively cleaned transcription audio.
    if cfg.diarization_audio_source != "raw":
        print(
            f"WARNING: diarization source '{cfg.diarization_audio_source}' is retained "
            "for CLI compatibility but ignored; using prepared original audio."
        )

    diarizer = Diarizer(
        hf_token=cfg.hf_token,
        device=cfg.whisper_device,
        clustering_threshold=cfg.diarization_clustering_threshold,
        clustering_fa=cfg.diarization_clustering_fa,
        clustering_fb=cfg.diarization_clustering_fb,
        segmentation_min_duration_off=cfg.diarization_segmentation_min_duration_off,
        embedding_validation=cfg.diarization_embedding_validation,
        embedding_min_similarity=cfg.diarization_embedding_min_similarity,
        embedding_min_margin=cfg.diarization_embedding_min_margin,
        embedding_max_assignment_gap=cfg.diarization_embedding_max_assignment_gap,
    )
    print(f"Diarizing on prepared original: {diarize_audio_path}  "
          f"(transcription will use: {audio_path})")
    diarization_segments = diarizer.run(
        diarize_audio_path,
        num_speakers=cfg.num_speakers,
        min_speakers=cfg.min_speakers,
        max_speakers=cfg.max_speakers,
        min_segment_duration=cfg.diarization_min_segment_duration,
        verbose=cfg.diarization_verbose,
    )

    # 4. Transcription - what was said, with word-level timestamps.
    transcriber = Transcriber(cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute_type)
    word_items, info = transcriber.transcribe(
        audio_path,
        language=cfg.language,
        beam_size=cfg.beam_size,
        vad_min_silence_ms=cfg.vad_min_silence_ms,
        condition_on_previous_text=cfg.condition_on_previous_text,
    )

    # 5. Alignment - map each word to a speaker.
    word_items = assign_speakers_to_words(word_items, diarization_segments)

    # Speaker-switch smoothing runs after alignment and before SRT generation.
    if cfg.enable_label_smoothing:
        word_items = smooth_speaker_labels(
            word_items,
            min_majority_ratio=cfg.smoothing_min_majority_ratio,
            protect_confidence=cfg.speaker_switch_protect_confidence,
            min_duration=cfg.speaker_switch_min_duration,
            context_words=cfg.speaker_switch_context_words,
            context_seconds=cfg.speaker_switch_context_seconds,
        )

    # 6. SRT generation.
    blocks = build_srt_blocks(word_items, cfg.max_line_duration, cfg.max_word_gap)
    write_srt(blocks, cfg.output_srt)


if __name__ == "__main__":
    cfg = Config()
    args = parse_args(cfg)

    cfg.input_audio = args.input
    cfg.output_srt = args.output
    apply_speaker_count_args(cfg, args)
    cfg.language = args.language
    cfg.whisper_device = args.device
    cfg.diarization_min_segment_duration = args.min_segment_duration
    cfg.enable_label_smoothing = not args.no_smoothing
    cfg.diarization_audio_source = args.diarize_source
    cfg.diarization_verbose = args.verbose_diarization
    cfg.diarization_clustering_threshold = args.clustering_threshold
    cfg.diarization_clustering_fa = args.clustering_fa
    cfg.diarization_clustering_fb = args.clustering_fb
    cfg.diarization_embedding_validation = not args.no_embedding_validation
    cfg.diarization_embedding_min_similarity = args.embedding_min_similarity
    cfg.diarization_embedding_min_margin = args.embedding_min_margin
    cfg.diarization_embedding_max_assignment_gap = args.embedding_max_assignment_gap
    if args.no_vocal_separation:
        cfg.do_vocal_separation = False
    if args.no_denoise:
        cfg.do_denoise = False

    run_pipeline(cfg)
