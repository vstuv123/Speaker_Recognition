"""
Benchmarks a speaker embedding model on identification quality.

Requires a test set laid out as:
  benchmark_clips/
    Feroze Khan/
      clip1.wav
      clip2.wav
      ...
    Sajal Aly/
      clip1.wav
      ...

Uses leave-one-out: for each clip, extract its embedding, then compare
against every OTHER clip (same script works whether clips came from
enrollment sources or held-out drama scenes -- just point --clips-dir
at different folders to test each condition separately).
"""
import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np

from embedder import EcapaEmbedder  # swap for WeSpeaker backend to compare


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_clips(clips_dir, embedder):
    """Returns {actor_name: [(clip_path, embedding), ...]}"""
    data = defaultdict(list)
    for actor_name in sorted(os.listdir(clips_dir)):
        actor_dir = os.path.join(clips_dir, actor_name)
        if not os.path.isdir(actor_dir):
            continue
        for filename in sorted(os.listdir(actor_dir)):
            if not filename.lower().endswith(".wav"):
                continue
            path = os.path.join(actor_dir, filename)
            emb = embedder.extract_from_file(path)
            data[actor_name].append((path, emb))
    return data


def pairwise_scores(data):
    """
    All same-speaker pairs and all different-speaker pairs, scored by
    cosine similarity. This is the raw material for every metric below.
    """
    same_scores, diff_scores = [], []
    same_pairs, diff_pairs = [], []

    all_clips = [(actor, path, emb) for actor, clips in data.items() for path, emb in clips]

    for (actor_a, path_a, emb_a), (actor_b, path_b, emb_b) in itertools.combinations(all_clips, 2):
        score = cosine_similarity(emb_a, emb_b)
        if actor_a == actor_b:
            same_scores.append(score)
            same_pairs.append((path_a, path_b, score))
        else:
            diff_scores.append(score)
            diff_pairs.append((path_a, path_b, score))

    return same_scores, diff_scores, same_pairs, diff_pairs


def compute_eer(same_scores, diff_scores):
    """
    Sweeps thresholds, finds where false-accept-rate == false-reject-rate.
    Standard speaker-verification metric -- lower EER is better.
    """
    all_scores = sorted(set(same_scores + diff_scores))
    best_threshold, best_gap, best_eer = None, float("inf"), 1.0

    for threshold in all_scores:
        far = sum(s >= threshold for s in diff_scores) / max(len(diff_scores), 1)
        frr = sum(s < threshold for s in same_scores) / max(len(same_scores), 1)
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_threshold, best_eer = gap, threshold, (far + frr) / 2

    return best_threshold, best_eer

def _far_frr_gap(threshold, same_scores, diff_scores):
    far = sum(s >= threshold for s in diff_scores) / max(len(diff_scores), 1)
    frr = sum(s < threshold for s in same_scores) / max(len(same_scores), 1)
    return far - frr


def threshold_sweep(same_scores, diff_scores, steps=41):
    """Precision/recall/F1 at each threshold -- for picking a production value."""
    results = []
    for threshold in np.linspace(0.0, 1.0, steps):
        tp = sum(s >= threshold for s in same_scores)
        fn = sum(s < threshold for s in same_scores)
        fp = sum(s >= threshold for s in diff_scores)
        tn = sum(s < threshold for s in diff_scores)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        results.append({
            "threshold": round(float(threshold), 3),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_accepts": fp,
            "false_rejects": fn,
        })
    return results


def confusion_matrix(data, threshold):
    """
    For each actor's clips, which OTHER actor do they get closest to
    (above threshold)? Surfaces systematic actor-pair confusion.
    """
    actors = list(data.keys())
    confusions = defaultdict(lambda: defaultdict(int))

    for actor_a in actors:
        for path_a, emb_a in data[actor_a]:
            best_actor, best_score = None, -1.0
            for actor_b in actors:
                if actor_b == actor_a:
                    continue
                for path_b, emb_b in data[actor_b]:
                    score = cosine_similarity(emb_a, emb_b)
                    if score > best_score:
                        best_score, best_actor = score, actor_b
            if best_score >= threshold:
                confusions[actor_a][best_actor] += 1

    return {a: dict(b) for a, b in confusions.items()}


def self_consistency(data):
    """
    Same-actor clip-to-clip similarity -- low values mean that actor's
    OWN enrollment clips disagree with each other, which will hurt
    identification regardless of how good the model is.
    """
    results = {}
    for actor, clips in data.items():
        if len(clips) < 2:
            results[actor] = {"note": "only 1 clip -- cannot measure", "mean_similarity": None}
            continue
        scores = [
            cosine_similarity(emb_a, emb_b)
            for (_, emb_a), (_, emb_b) in itertools.combinations(clips, 2)
        ]
        results[actor] = {
            "mean_similarity": round(float(np.mean(scores)), 4),
            "min_similarity": round(float(np.min(scores)), 4),
            "num_clip_pairs": len(scores),
        }
    return results


def run_benchmark(clips_dir, output_path, device="cuda", model_name="ecapa"):
    print(f"Loading model: {model_name}")
    embedder = EcapaEmbedder(device=device)  # swap backend here per model

    print(f"Extracting embeddings from: {clips_dir}")
    data = load_clips(clips_dir, embedder)
    total_clips = sum(len(v) for v in data.values())
    print(f"Loaded {total_clips} clips across {len(data)} actors")

    same_scores, diff_scores, same_pairs, diff_pairs = pairwise_scores(data)
    eer_threshold, eer = compute_eer(same_scores, diff_scores)
    sweep = threshold_sweep(same_scores, diff_scores)
    consistency = self_consistency(data)
    confusions = confusion_matrix(data, threshold=eer_threshold)

    best_f1_row = max(sweep, key=lambda r: r["f1"])

    report = {
        "model": model_name,
        "num_actors": len(data),
        "num_clips": total_clips,
        "same_speaker_pairs": len(same_scores),
        "different_speaker_pairs": len(diff_scores),
        "same_speaker_similarity": {
            "mean": round(float(np.mean(same_scores)), 4),
            "std": round(float(np.std(same_scores)), 4),
            "min": round(float(np.min(same_scores)), 4),
        },
        "different_speaker_similarity": {
            "mean": round(float(np.mean(diff_scores)), 4),
            "std": round(float(np.std(diff_scores)), 4),
            "max": round(float(np.max(diff_scores)), 4),
        },
        "separation_gap": round(
            float(np.mean(same_scores)) - float(np.mean(diff_scores)), 4
        ),
        "eer": round(eer, 4),
        "eer_threshold": round(eer_threshold, 4),
        "best_f1_operating_point": best_f1_row,
        "self_consistency_per_actor": consistency,
        "confusions_at_eer_threshold": confusions,
        "threshold_sweep": sweep,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- Summary ---")
    print(f"Same-speaker similarity:      {report['same_speaker_similarity']['mean']:.3f} +/- {report['same_speaker_similarity']['std']:.3f}")
    print(f"Different-speaker similarity: {report['different_speaker_similarity']['mean']:.3f} +/- {report['different_speaker_similarity']['std']:.3f}")
    print(f"Separation gap:               {report['separation_gap']:.3f}  (higher = cleaner separation)")
    print(f"EER:                          {report['eer']*100:.2f}%  @ threshold {eer_threshold:.3f}")
    print(f"Best F1 operating point:      {best_f1_row}")
    if confusions:
        print("\nConfusions at EER threshold:")
        for actor, confused_with in confusions.items():
            print(f"  {actor} -> {confused_with}")
    print(f"\nFull report saved: {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", default="benchmark_clips")
    parser.add_argument("--output", default="benchmark_report.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="ecapa")
    args = parser.parse_args()
    run_benchmark(args.clips_dir, args.output, args.device, args.model_name)