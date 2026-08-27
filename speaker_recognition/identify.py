"""
Matches diarized SPEAKER_XX clusters against embeddings.json using
cosine similarity.
"""
import json
from collections import defaultdict

import numpy as np
import soundfile as sf

from .wespeaker_embedder import WeSpeakerEmbedder

DEFAULT_THRESHOLD = 0.50 # similarity threshold for accepting a match
MAX_CLUSTER_SECONDS = 30.0  # cap audio per cluster -- more isn't always better


def cosine_similarity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_embeddings_db(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {name: [np.asarray(v) for v in vectors] for name, vectors in raw.items()}


def collect_speaker_audio(audio_path, diarization_segments, max_seconds=MAX_CLUSTER_SECONDS):
    """Concatenate each SPEAKER_XX's longest segments (up to max_seconds)."""
    info = sf.info(audio_path)
    sr = info.samplerate

    by_speaker = defaultdict(list)
    for seg in diarization_segments:
        by_speaker[seg["speaker"]].append(seg)

    speaker_audio = {}
    with sf.SoundFile(audio_path) as f:
        for speaker, segments in by_speaker.items():
            if speaker == "UNKNOWN":  # diarization's own rejection track -- skip
                continue
            chunks, collected = [], 0.0
            for seg in sorted(segments, key=lambda s: -(s["end"] - s["start"])):
                if collected >= max_seconds:
                    break
                start_frame, end_frame = int(seg["start"] * sr), int(seg["end"] * sr)
                f.seek(start_frame)
                chunk = f.read(end_frame - start_frame, dtype="float32", always_2d=True)
                mono = chunk.mean(axis=1)
                chunks.append(mono)
                collected += len(mono) / sr
            if chunks:
                speaker_audio[speaker] = (np.concatenate(chunks), sr)
    return speaker_audio


def identify_speakers(audio_path, diarization_segments, embeddings_db_path,
                       threshold=DEFAULT_THRESHOLD, device="cuda", model_dir="models/samresnet100_voxblink2"):
    """Returns {"SPEAKER_00": "Feroze Khan", "SPEAKER_01": "UNKNOWN", ...}"""
    embedder = WeSpeakerEmbedder(model_dir=model_dir, device=device)
    db = load_embeddings_db(embeddings_db_path)
    speaker_audio = collect_speaker_audio(audio_path, diarization_segments)

    mapping = {}
    for speaker_label, (waveform, sr) in speaker_audio.items():
        query = embedder.extract(waveform, sr)

        best_name, best_score = "UNKNOWN", -1.0
        for actor_name, actor_embeddings in db.items():
            # Compare against every enrolled clip individually rather than
            # a pre-averaged centroid -- keeps per-clip variance usable.
            for actor_embedding in actor_embeddings:
                score = cosine_similarity(query, actor_embedding)
                if score > best_score:
                    best_score, best_name = score, actor_name

        if best_score >= threshold:
            mapping[speaker_label] = best_name
            print(f"  {speaker_label} -> {best_name} (similarity={best_score:.3f})")
        else:
            mapping[speaker_label] = "UNKNOWN"
            print(f"  {speaker_label} -> UNKNOWN (closest={best_name}, {best_score:.3f} < {threshold})")

    return mapping