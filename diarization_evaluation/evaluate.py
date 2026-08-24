"""Unsupervised benchmark for Community-1 diarization parameters.

This script is intentionally separate from the production pipeline. It does
not modify Config or apply the winning values to normal runs.
"""

import argparse
import copy
import csv
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyannote.audio
import soundfile as sf
import torch
from pyannote.audio import Pipeline

from alignment.aligner import assign_speakers_to_words, smooth_speaker_labels
from config import Config
from subtitles.srt_writer import build_srt_blocks, write_srt
from transcription.transcriber import Transcriber


MODEL_NAME = "pyannote/speaker-diarization-community-1"
THRESHOLDS = (0.55, 0.60, 0.65, 0.70)
MIN_DURATIONS_OFF = (0.00, 0.10, 0.20, 0.30)
SHORT_LIMITS = (0.25, 0.50, 1.00)
EPSILON = 1e-9


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", default="audio.wav")
    parser.add_argument("--output-dir", default="diarization_evaluation")
    parser.add_argument(
        "--transcription-audio",
        default=None,
        help="Audio for the one-time Whisper pass. Defaults to workdir/cleaned.wav when present, otherwise --audio.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def _annotation_segments(annotation):
    return [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        for turn, speaker in annotation
    ]


def _union_duration(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end + EPSILON:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def detect_overlap_regions(segments):
    """Derive regions with at least two unique active speaker identities."""
    events = []
    for segment in segments:
        if segment["end"] > segment["start"]:
            events.append((segment["start"], 1, segment["speaker"]))
            events.append((segment["end"], -1, segment["speaker"]))
    events.sort(key=lambda item: item[0])
    active = Counter()
    result = []
    previous = events[0][0] if events else 0.0
    index = 0
    while index < len(events):
        timestamp = events[index][0]
        speakers = sorted(speaker for speaker, count in active.items() if count > 0)
        if len(speakers) >= 2 and timestamp - previous > EPSILON:
            region = {"start": previous, "end": timestamp, "speakers": speakers}
            region["duration"] = region["end"] - region["start"]
            result.append(region)
        while index < len(events) and abs(events[index][0] - timestamp) <= EPSILON:
            _, delta, speaker = events[index]
            active[speaker] += delta
            if active[speaker] <= 0:
                del active[speaker]
            index += 1
        previous = timestamp
    return result


def diarization_diagnostics(exclusive_segments, regular_segments, audio_duration):
    segments = sorted(exclusive_segments, key=lambda item: (item["start"], item["end"]))
    durations = [max(0.0, item["end"] - item["start"]) for item in segments]
    per_speaker = defaultdict(float)
    turns_per_speaker = Counter()
    for segment, duration in zip(segments, durations):
        per_speaker[segment["speaker"]] += duration
        turns_per_speaker[segment["speaker"]] += 1

    speech_duration = _union_duration([(item["start"], item["end"]) for item in segments])
    gaps = []
    previous_end = None
    for segment in segments:
        if previous_end is not None and segment["start"] - previous_end > EPSILON:
            gaps.append(segment["start"] - previous_end)
        previous_end = max(previous_end or segment["end"], segment["end"])

    overlaps = detect_overlap_regions(regular_segments)
    overlap_duration = sum(item["duration"] for item in overlaps)
    mean_duration = statistics.mean(durations) if durations else 0.0
    median_duration = statistics.median(durations) if durations else 0.0
    speaker_values = list(per_speaker.values())
    balance_cv = (
        statistics.pstdev(speaker_values) / statistics.mean(speaker_values)
        if len(speaker_values) > 1 and statistics.mean(speaker_values) > 0 else 0.0
    )
    speech_minutes = max(speech_duration / 60.0, EPSILON)

    return {
        "detected_speakers": len(per_speaker),
        "speaker_turns": len(segments),
        "total_speech_duration": speech_duration,
        "average_turn_duration": mean_duration,
        "median_turn_duration": median_duration,
        "extremely_short_turns": sum(duration < 0.25 for duration in durations),
        "short_turns_under_0_25": sum(duration < 0.25 for duration in durations),
        "short_turns_under_0_50": sum(duration < 0.50 for duration in durations),
        "short_turns_under_1_00": sum(duration < 1.00 for duration in durations),
        "gap_count": len(gaps),
        "gap_duration": sum(gaps),
        "overlap_region_count": len(overlaps),
        "overlap_duration": overlap_duration,
        "short_overlap_regions": sum(item["duration"] < 0.25 for item in overlaps),
        "fragmentation_per_speech_minute": len(segments) / speech_minutes,
        "speaker_duration_balance_cv": balance_cv,
        "minor_speaker_count": sum(
            duration / max(audio_duration, EPSILON) < 0.01 for duration in speaker_values
        ),
        "duration_per_speaker": dict(sorted(per_speaker.items())),
        "turns_per_speaker": dict(sorted(turns_per_speaker.items())),
        "audio_percentage_per_speaker": {
            speaker: duration / max(audio_duration, EPSILON) * 100.0
            for speaker, duration in sorted(per_speaker.items())
        },
        "overlap_regions": overlaps,
    }


def _word_runs(words):
    runs = []
    for word in words:
        speaker = word.get("speaker", "UNKNOWN")
        if runs and runs[-1]["speaker"] == speaker:
            runs[-1]["end"] = word["end"]
            runs[-1]["words"] += 1
        else:
            runs.append({"speaker": speaker, "start": word["start"], "end": word["end"], "words": 1})
    return runs


def word_diagnostics(words):
    if not words:
        return {
            "word_count": 0, "word_switches": 0, "isolated_one_word_assignments": 0,
            "rapid_word_switches": 0, "suspicious_aba_patterns": 0,
            "average_words_per_speaker_segment": 0.0, "unknown_word_percentage": 0.0,
            "word_percentage_per_speaker": {},
        }
    labels = [word.get("speaker", "UNKNOWN") for word in words]
    switches = [index for index in range(1, len(words)) if labels[index] != labels[index - 1]]
    isolated = sum(
        labels[index - 1] == labels[index + 1] != labels[index]
        for index in range(1, len(labels) - 1)
    )
    aba = isolated
    rapid = sum(
        max(0.0, words[index]["start"] - words[index - 1]["end"]) < 0.25
        for index in switches
    )
    runs = _word_runs(words)
    counts = Counter(labels)
    return {
        "word_count": len(words),
        "word_switches": len(switches),
        "isolated_one_word_assignments": isolated,
        "rapid_word_switches": rapid,
        "suspicious_aba_patterns": aba,
        "average_words_per_speaker_segment": statistics.mean(run["words"] for run in runs),
        "unknown_word_percentage": counts.get("UNKNOWN", 0) / len(words) * 100.0,
        "word_percentage_per_speaker": {
            speaker: count / len(words) * 100.0 for speaker, count in sorted(counts.items())
        },
    }


PENALTY_WEIGHTS = {
    "fragmentation_per_speech_minute": 20.0,
    "short_turn_rate": 17.5,
    "word_switch_rate": 17.5,
    "rapid_word_switch_rate": 12.5,
    "isolated_word_rate": 10.0,
    "aba_rate": 7.5,
    "speaker_duration_balance_cv": 5.0,
    "minor_speaker_rate": 5.0,
    "short_overlap_rate": 2.5,
    "unknown_word_rate": 2.5,
}


def _penalty_values(result):
    turns = max(result["speaker_turns"], 1)
    words = max(result["word_count"], 1)
    speakers = max(result["detected_speakers"], 1)
    speech_minutes = max(result["total_speech_duration"] / 60.0, EPSILON)
    return {
        "fragmentation_per_speech_minute": result["fragmentation_per_speech_minute"],
        "short_turn_rate": result["extremely_short_turns"] / turns,
        "word_switch_rate": result["word_switches"] / words,
        "rapid_word_switch_rate": result["rapid_word_switches"] / words,
        "isolated_word_rate": result["isolated_one_word_assignments"] / words,
        "aba_rate": result["suspicious_aba_patterns"] / words,
        "speaker_duration_balance_cv": result["speaker_duration_balance_cv"],
        "minor_speaker_rate": result["minor_speaker_count"] / speakers,
        "short_overlap_rate": result["short_overlap_regions"] / speech_minutes,
        "unknown_word_rate": result["unknown_word_percentage"] / 100.0,
    }


def rank_results(results):
    """Assign transparent relative heuristic scores (higher is better)."""
    if not results:
        return results
    raw = [_penalty_values(result) for result in results]
    ranges = {}
    for metric in PENALTY_WEIGHTS:
        values = [item[metric] for item in raw]
        ranges[metric] = (min(values), max(values))
    for result, penalties in zip(results, raw):
        normalized = {}
        weighted = {}
        for metric, weight in PENALTY_WEIGHTS.items():
            low, high = ranges[metric]
            normalized[metric] = 0.0 if high - low <= EPSILON else (penalties[metric] - low) / (high - low)
            weighted[metric] = normalized[metric] * weight
        result["penalty_values"] = penalties
        result["normalized_penalties"] = normalized
        result["weighted_penalties"] = weighted
        result["score"] = max(0.0, 100.0 - sum(weighted.values()))
    return sorted(results, key=lambda item: (-item["score"], item["configuration_id"]))


def _audio_as_waveform(path):
    waveform, sample_rate = sf.read(path, always_2d=True)
    return {
        "waveform": torch.from_numpy(np.asarray(waveform, dtype=np.float32).T),
        "sample_rate": int(sample_rate),
    }


def _word_cache_key(path, cfg):
    stat = Path(path).stat()
    return {
        "path": str(Path(path).resolve()), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "model": cfg.whisper_model,
        "language": cfg.language,
    }


def load_or_transcribe(path, output_dir, cfg):
    cache_path = output_dir / "whisper_words.json"
    key = _word_cache_key(path, cfg)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == key:
            print(f"Reusing {len(cached['words'])} cached Whisper words from {cache_path}")
            return cached["words"]
    transcriber = Transcriber(cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute_type)
    words, _ = transcriber.transcribe(
        path, language=cfg.language, beam_size=cfg.beam_size,
        vad_min_silence_ms=cfg.vad_min_silence_ms,
        condition_on_previous_text=cfg.condition_on_previous_text,
    )
    cache_path.write_text(json.dumps({"cache_key": key, "words": words}, indent=2), encoding="utf-8")
    return words


def build_experiment_configurations():
    return [
        {"experiment": 1, "threshold": value, "min_duration_off": 0.0,
         "speaker_mode": "automatic", "min_speakers": 2, "max_speakers": 7}
        for value in THRESHOLDS
    ]


def _configuration_key(config):
    return (
        config["threshold"], config["min_duration_off"], config["speaker_mode"],
        config.get("num_speakers"), config.get("min_speakers"), config.get("max_speakers"),
    )


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_configuration(pipeline, defaults, waveform, words, config, config_dir, audio_duration, cfg):
    params = copy.deepcopy(defaults)
    params["clustering"]["threshold"] = config["threshold"]
    params["segmentation"]["min_duration_off"] = config["min_duration_off"]
    pipeline.instantiate(params)
    kwargs = (
        {"num_speakers": config["num_speakers"]}
        if config["speaker_mode"] == "exact"
        else {"min_speakers": config["min_speakers"], "max_speakers": config["max_speakers"]}
    )
    started = time.perf_counter()
    output = pipeline(waveform, **kwargs)
    runtime = time.perf_counter() - started
    exclusive = _annotation_segments(output.exclusive_speaker_diarization)
    regular = _annotation_segments(output.speaker_diarization)

    aligned = assign_speakers_to_words(copy.deepcopy(words), exclusive)
    if cfg.enable_label_smoothing:
        aligned = smooth_speaker_labels(
            aligned, min_majority_ratio=cfg.smoothing_min_majority_ratio,
            protect_confidence=cfg.speaker_switch_protect_confidence,
            min_duration=cfg.speaker_switch_min_duration,
            context_words=cfg.speaker_switch_context_words,
            context_seconds=cfg.speaker_switch_context_seconds,
        )
    blocks = build_srt_blocks(aligned, cfg.max_line_duration, cfg.max_word_gap)
    write_srt(blocks, str(config_dir / "output.srt"))

    diagnostics = diarization_diagnostics(exclusive, regular, audio_duration)
    diagnostics.update(word_diagnostics(aligned))
    diagnostics.update({
        "runtime_seconds": runtime,
        "configuration_id": config["configuration_id"],
        "experiment": config["experiment"],
    })
    _write_json(config_dir / "configuration.json", config)
    _write_json(config_dir / "diarization.json", {"exclusive": exclusive, "regular": regular})
    _write_json(config_dir / "aligned_words.json", aligned)
    _write_json(config_dir / "diagnostics.json", diagnostics)
    return {**config, **diagnostics}


def _next_id(number):
    return f"config_{number:02d}"


def _flat_result(result):
    return {
        key: value for key, value in result.items()
        if not isinstance(value, (dict, list))
    }


def write_results(output_dir, ranked, metadata):
    payload = {"evaluation": "unsupervised_heuristic", "metadata": metadata,
               "scoring_weights": PENALTY_WEIGHTS, "results": ranked}
    _write_json(output_dir / "results.json", payload)
    rows = [_flat_result(result) for result in ranked]
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "UNSUPERVISED / HEURISTIC DIARIZATION EVALUATION",
        "No ground-truth speaker annotation was available; scores measure stability, not correctness.",
        "",
        "Configuration | Speakers | Turns | Short Turns | Fragmentation | Word Switches | Score",
        "-" * 96,
    ]
    for result in ranked:
        lines.append(
            f"{result['configuration_id']} | {result['detected_speakers']} | "
            f"{result['speaker_turns']} | {result['extremely_short_turns']} | "
            f"{result['fragmentation_per_speech_minute']:.2f} | "
            f"{result['word_switches']} | {result['score']:.2f}"
        )
    if ranked:
        stable = min(ranked, key=lambda item: (item["word_switches"], item["speaker_turns"]))
        conservative = min(ranked, key=lambda item: (item["detected_speakers"], item["speaker_turns"]))
        lines.extend([
            "", f"BEST CONFIGURATION: {ranked[0]['configuration_id']}",
            f"MOST STABLE CONFIGURATION: {stable['configuration_id']}",
            f"MOST CONSERVATIVE CONFIGURATION: {conservative['configuration_id']}",
            "", "Recommendation must be reviewed against the audio before production use.",
        ])
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def create_plots(output_dir, ranked):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, RuntimeError) as exc:
        print(f"matplotlib unavailable; skipping plots: {exc}")
        return
    labels = [item["configuration_id"] for item in ranked]
    plots = [
        ("detected_speakers", "Detected speakers", "speakers.png"),
        ("speaker_turns", "Speaker turns", "turns.png"),
        ("extremely_short_turns", "Turns under 0.25 s", "short_turns.png"),
        ("fragmentation_per_speech_minute", "Turns per speech minute", "fragmentation.png"),
        ("score", "Heuristic score", "ranking.png"),
    ]
    for metric, title, filename in plots:
        plt.figure(figsize=(10, 4))
        plt.bar(labels, [item[metric] for item in ranked])
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=140)
        plt.close()


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    cfg = Config()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.whisper_device = device
    if device == "cpu" and cfg.whisper_compute_type == "float16":
        cfg.whisper_compute_type = "int8"

    audio_path = Path(args.audio)
    output_dir = Path(args.output_dir)
    configurations_dir = output_dir / "configurations"
    configurations_dir.mkdir(parents=True, exist_ok=True)
    audio_info = sf.info(str(audio_path))
    if args.transcription_audio:
        transcription_audio = Path(args.transcription_audio)
    else:
        candidates = [
            Path("workdir/cleaned.wav"),
            Path("workdir/separated/htdemucs/audio/vocals.wav"),
            audio_path,
        ]
        transcription_audio = audio_path
        for candidate in candidates:
            if candidate.exists() and abs(sf.info(str(candidate)).duration - audio_info.duration) <= 0.25:
                transcription_audio = candidate
                break
        if Path("workdir/cleaned.wav").exists() and transcription_audio != Path("workdir/cleaned.wav"):
            print(
                "WARNING: workdir/cleaned.wav duration does not match audio.wav; "
                f"using {transcription_audio} for aligned Whisper timestamps instead."
            )
    if not transcription_audio.exists():
        raise FileNotFoundError(f"Transcription audio not found: {transcription_audio}")

    words = load_or_transcribe(str(transcription_audio), output_dir, cfg)
    token = cfg.hf_token
    if not token:
        raise RuntimeError("HF_TOKEN is required to load Community-1")
    pipeline = Pipeline.from_pretrained(MODEL_NAME, token=token)
    defaults = pipeline.parameters(instantiated=True)
    pipeline.to(torch.device(device))
    waveform = _audio_as_waveform(str(audio_path))

    all_results = []
    seen = set()
    number = 1

    def run_configs(configs):
        nonlocal number
        new_results = []
        for config in configs:
            key = _configuration_key(config)
            if key in seen:
                existing = next(item for item in all_results if _configuration_key(item) == key)
                new_results.append(existing)
                continue
            seen.add(key)
            config = dict(config, configuration_id=_next_id(number))
            number += 1
            config_dir = configurations_dir / config["configuration_id"]
            config_dir.mkdir(exist_ok=True)
            print(f"\nEvaluating {config['configuration_id']}: {config}")
            result = evaluate_configuration(
                pipeline, defaults, waveform, words, config, config_dir,
                audio_info.duration, cfg,
            )
            all_results.append(result)
            new_results.append(result)
        return new_results

    experiment_1 = run_configs(build_experiment_configurations())
    best_threshold = rank_results(copy.deepcopy(experiment_1))[0]["threshold"]
    experiment_2 = run_configs([
        {"experiment": 2, "threshold": best_threshold, "min_duration_off": value,
         "speaker_mode": "automatic", "min_speakers": 2, "max_speakers": 7}
        for value in MIN_DURATIONS_OFF
    ])
    best_min_duration = rank_results(copy.deepcopy(experiment_2))[0]["min_duration_off"]
    run_configs([
        {"experiment": 3, "threshold": best_threshold, "min_duration_off": best_min_duration,
         "speaker_mode": "automatic", "min_speakers": 2, "max_speakers": 7},
        {"experiment": 3, "threshold": best_threshold, "min_duration_off": best_min_duration,
         "speaker_mode": "exact", "num_speakers": 7},
    ])

    ranked = rank_results(all_results)
    metadata = {
        "model": MODEL_NAME, "pyannote_audio_version": pyannote.audio.__version__,
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "device": device, "audio_filename": str(audio_path),
        "audio_sample_rate": audio_info.samplerate, "audio_duration": audio_info.duration,
        "transcription_audio": str(transcription_audio), "whisper_model": cfg.whisper_model,
        "community_defaults": defaults, "embedding_exclude_overlap": True,
        "speaker_count_ground_truth_available": False,
    }
    write_results(output_dir, ranked, metadata)
    if not args.skip_plots:
        create_plots(output_dir, ranked)
    print(f"\nArtifacts saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
