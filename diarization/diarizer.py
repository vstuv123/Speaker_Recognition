# """
# Speaker diarization using pyannote/speaker-diarization-community-1.
# """

# import numpy as np
# import soundfile as sf
# import torch
# from pyannote.audio import Pipeline


# class Diarizer:
#     def __init__(self, hf_token, device=None):
#         if not hf_token:
#             raise RuntimeError(
#                 "HF_TOKEN is not set.\n\n"
#                 "PowerShell:\n"
#                 '$env:HF_TOKEN="hf_your_token_here"\n'
#             )

#         print("Loading pyannote Community-1 diarization pipeline...")
#         self.pipeline = Pipeline.from_pretrained(
#             "pyannote/speaker-diarization-community-1",
#             token=hf_token,
#         )

#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"

#         if device == "cuda" and torch.cuda.is_available():
#             print("CUDA available. Moving pyannote pipeline to GPU...")
#             self.pipeline.to(torch.device("cuda"))
#         else:
#             print("Pyannote will run on CPU.")

#     @staticmethod
#     def _load_audio_as_waveform(audio_path):
#         waveform, sample_rate = sf.read(audio_path, always_2d=True)
#         waveform = np.asarray(waveform, dtype=np.float32)
#         waveform = waveform.T
#         return {"waveform": torch.from_numpy(waveform), "sample_rate": int(sample_rate)}

#     @staticmethod
#     def _merge_short_segments(segments, min_duration):
#         """
#         Merge diarization segments shorter than `min_duration` seconds into
#         the preceding segment. Very short segments (a breath, a mic pop, a
#         clustering boundary artifact) are usually noise rather than a real
#         turn; left alone they fragment one real speaker's audio across
#         multiple tiny segments and add churn for whichever speaker label
#         gets assigned next. Off by default (min_duration=0) -- only turn
#         this on if the diagnostic printout below shows many sub-0.3s
#         segments, since it does throw away genuinely short interjections
#         too.
#         """
#         if min_duration <= 0 or len(segments) < 2:
#             return segments

#         segments = sorted(segments, key=lambda s: s["start"])
#         merged = [dict(segments[0])]

#         for seg in segments[1:]:
#             duration = seg["end"] - seg["start"]
#             prev = merged[-1]
#             if duration < min_duration and (seg["start"] - prev["end"]) < 1.0:
#                 prev["end"] = max(prev["end"], seg["end"])
#             else:
#                 merged.append(dict(seg))

#         return merged

#     def run(self, audio_file, num_speakers=None, min_speakers=None, max_speakers=None,
#             min_segment_duration=0.0):
#         kwargs = {}
#         if num_speakers is not None:
#             print(f"Using exact number of speakers: {num_speakers}")
#             kwargs["num_speakers"] = num_speakers
#         else:
#             print(f"Using speaker range: {min_speakers}-{max_speakers}")
#             kwargs["min_speakers"] = min_speakers
#             kwargs["max_speakers"] = max_speakers

#         try:
#             output = self.pipeline(audio_file, **kwargs)
#         except RuntimeError as exc:
#             if "torchcodec is not available" in str(exc) or "Cannot read audio file" in str(exc):
#                 print("TorchCodec unavailable; loading audio into memory as waveform dictionary...")
#                 audio = self._load_audio_as_waveform(audio_file)
#                 output = self.pipeline(audio, **kwargs)
#             else:
#                 raise

#         # Community-1 provides exclusive speaker diarization, which is
#         # useful when combining diarization with Whisper word timestamps.
#         diarization = getattr(output, "exclusive_speaker_diarization", output.speaker_diarization)

#         segments = []
#         for turn, speaker in diarization:
#             segments.append({
#                 "start": float(turn.start),
#                 "end": float(turn.end),
#                 "speaker": speaker,
#             })

#         segments.sort(key=lambda s: s["start"])

#         # --- Diagnostics ---
#         # If num_speakers was forced (e.g. 7) but one label shows up with
#         # only a couple of tiny segments here, that's a strong sign the
#         # true speaker count doesn't match the forced value, and the
#         # clustering was forced to split or merge real speakers to comply.
#         # Re-run with num_speakers=None and a min/max range instead.
#         per_speaker = {}
#         for s in segments:
#             stats = per_speaker.setdefault(s["speaker"], {"count": 0, "duration": 0.0})
#             stats["count"] += 1
#             stats["duration"] += s["end"] - s["start"]

#         print(f"Detected diarization segments: {len(segments)}")
#         print(f"Distinct speaker labels: {len(per_speaker)}")
#         for spk, stats in sorted(per_speaker.items()):
#             print(f"  {spk}: {stats['count']} segments, {stats['duration']:.1f}s total")

#         if min_segment_duration > 0:
#             before = len(segments)
#             segments = self._merge_short_segments(segments, min_segment_duration)
#             print(f"Merged short segments (<{min_segment_duration}s): {before} -> {len(segments)}")

#         return segments

"""
Speaker diarization using pyannote/speaker-diarization-community-1.
"""

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline
import copy


UNKNOWN_SPEAKER = "UNKNOWN"
EPSILON = 1e-12


def validate_embedding_assignments(
    embeddings,
    hard_clusters,
    centroids,
    segmentations,
    *,
    min_similarity,
    min_margin,
    max_assignment_gap,
):
    """Reject local speaker embeddings that do not support their VBx label.

    Community-1 normally assigns every active local speaker track to a
    centroid.  That forced assignment is useful for complete diarization, but
    it can make a weak or contaminated embedding look like a confident speaker
    decision.  This validator preserves good VBx assignments and routes weak
    evidence to one extra UNKNOWN cluster.

    ``min_margin`` applies when VBx selected the embedding's nearest centroid.
    When VBx intentionally selected another centroid to keep simultaneous
    local speakers distinct, ``max_assignment_gap`` allows a small constrained
    difference but rejects a clearly implausible assignment.
    """
    embeddings = np.asarray(embeddings)
    hard_clusters = np.asarray(hard_clusters).copy()
    centroids = np.asarray(centroids)
    segmentation_data = np.asarray(
        segmentations.data if hasattr(segmentations, "data") else segmentations
    )

    stats = {
        "checked": 0,
        "rejected": 0,
        "low_similarity": 0,
        "ambiguous": 0,
        "large_assignment_gap": 0,
    }
    rejected = np.zeros(hard_clusters.shape, dtype=bool)
    if centroids.size == 0 or centroids.shape[0] == 0:
        return hard_clusters, rejected, stats

    flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    flat_assignments = hard_clusters.reshape(-1)
    active = segmentation_data.sum(axis=1).reshape(-1) > 0
    finite = np.isfinite(flat_embeddings).all(axis=1)
    valid = active & finite & (flat_assignments >= 0)
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return hard_clusters, rejected, stats

    valid_embeddings = flat_embeddings[valid_indices]
    embedding_norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    nonzero = (embedding_norms[:, 0] > EPSILON)
    usable_centroids = centroid_norms[:, 0] > EPSILON
    if not np.any(usable_centroids):
        rejected.reshape(-1)[valid_indices] = True
        hard_clusters.reshape(-1)[valid_indices] = centroids.shape[0]
        stats.update(checked=len(valid_indices), rejected=len(valid_indices), low_similarity=len(valid_indices))
        return hard_clusters, rejected, stats

    normalized_embeddings = valid_embeddings / np.maximum(embedding_norms, EPSILON)
    normalized_centroids = centroids / np.maximum(centroid_norms, EPSILON)
    similarities = normalized_embeddings @ normalized_centroids.T
    similarities[:, ~usable_centroids] = -np.inf

    assigned = flat_assignments[valid_indices]
    row = np.arange(len(valid_indices))
    assigned_similarity = similarities[row, assigned]
    best = np.argmax(similarities, axis=1)
    best_similarity = similarities[row, best]

    competitors = similarities.copy()
    competitors[row, assigned] = -np.inf
    competitor_similarity = np.max(competitors, axis=1)
    assigned_margin = assigned_similarity - competitor_similarity
    assignment_gap = best_similarity - assigned_similarity

    low_similarity = (~nonzero) | (assigned_similarity < min_similarity)
    ambiguous = (best == assigned) & (assigned_margin < min_margin)
    large_gap = (best != assigned) & (assignment_gap > max_assignment_gap)
    reject_valid = low_similarity | ambiguous | large_gap

    rejected_flat = rejected.reshape(-1)
    rejected_flat[valid_indices[reject_valid]] = True
    hard_clusters.reshape(-1)[valid_indices[reject_valid]] = centroids.shape[0]
    stats.update(
        checked=int(len(valid_indices)),
        rejected=int(np.sum(reject_valid)),
        low_similarity=int(np.sum(low_similarity)),
        ambiguous=int(np.sum(ambiguous & ~low_similarity)),
        large_assignment_gap=int(
            np.sum(large_gap & ~low_similarity & ~ambiguous)
        ),
    )
    return hard_clusters, rejected, stats


class ConfidenceAwareClustering:
    """Add an UNKNOWN rejection cluster after Community-1 VBx clustering."""

    def __init__(
        self,
        base_clustering,
        *,
        min_similarity,
        min_margin,
        max_assignment_gap,
    ):
        self.base_clustering = base_clustering
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self.max_assignment_gap = max_assignment_gap
        self.last_stats = None

    def __call__(self, embeddings, segmentations=None, **kwargs):
        hard_clusters, soft_clusters, centroids = self.base_clustering(
            embeddings=embeddings,
            segmentations=segmentations,
            **kwargs,
        )
        if segmentations is None or centroids is None:
            return hard_clusters, soft_clusters, centroids

        hard_clusters, rejected, stats = validate_embedding_assignments(
            embeddings,
            hard_clusters,
            centroids,
            segmentations,
            min_similarity=self.min_similarity,
            min_margin=self.min_margin,
            max_assignment_gap=self.max_assignment_gap,
        )
        self.last_stats = stats
        if not stats["rejected"]:
            return hard_clusters, soft_clusters, centroids

        # A zero centroid is intentional: Diarizer.run uses it to identify the
        # extra output label and rename that label to UNKNOWN.
        unknown_index = centroids.shape[0]
        centroids = np.vstack([centroids, np.zeros((1, centroids.shape[1]))])
        if soft_clusters is not None:
            floor = float(np.nanmin(soft_clusters)) - 1.0
            soft_clusters = np.pad(
                soft_clusters,
                ((0, 0), (0, 0), (0, 1)),
                constant_values=floor,
            )
            soft_clusters[..., unknown_index][rejected] = float(np.nanmax(soft_clusters)) + 1.0
        return hard_clusters, soft_clusters, centroids


class Diarizer:
    def __init__(
        self,
        hf_token,
        device=None,
        clustering_threshold=None,
        clustering_fa=None,
        clustering_fb=None,
        segmentation_min_duration_off=None,
        embedding_validation=True,
        embedding_min_similarity=0.35,
        embedding_min_margin=0.05,
        embedding_max_assignment_gap=0.20,
    ):
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set.\n\n"
                "PowerShell:\n"
                '$env:HF_TOKEN="hf_your_token_here"\n'
            )
        if not -1.0 <= embedding_min_similarity <= 1.0:
            raise ValueError("embedding_min_similarity must be between -1 and 1")
        if embedding_min_margin < 0.0:
            raise ValueError("embedding_min_margin must be non-negative")
        if embedding_max_assignment_gap < 0.0:
            raise ValueError("embedding_max_assignment_gap must be non-negative")

        print("Loading pyannote Community-1 diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
        print("Pipeline hyperparameters:", self.pipeline.parameters(instantiated=True))
        default_params = self.pipeline.parameters(instantiated=True)
        print("Pipeline hyperparameters (defaults):", default_params)

        overrides = {}
        if clustering_threshold is not None:
            overrides["threshold"] = clustering_threshold
        if clustering_fa is not None:
            overrides["Fa"] = clustering_fa
        if clustering_fb is not None:
            overrides["Fb"] = clustering_fb

        if overrides or segmentation_min_duration_off is not None:
            new_params = copy.deepcopy(default_params)
            new_params.setdefault("clustering", {}).update(overrides)
            if segmentation_min_duration_off is not None:
                new_params.setdefault("segmentation", {})["min_duration_off"] = (
                    segmentation_min_duration_off
                )
            print(f"Overriding clustering params: {overrides}")
            if segmentation_min_duration_off is not None:
                print(
                    "Overriding segmentation.min_duration_off: "
                    f"{segmentation_min_duration_off}"
                )
            self.pipeline.instantiate(new_params)
            print("Pipeline hyperparameters (after override):",
                self.pipeline.parameters(instantiated=True))

        self.confidence_clustering = None
        if embedding_validation:
            self.confidence_clustering = ConfidenceAwareClustering(
                self.pipeline.clustering,
                min_similarity=embedding_min_similarity,
                min_margin=embedding_min_margin,
                max_assignment_gap=embedding_max_assignment_gap,
            )
            self.pipeline.clustering = self.confidence_clustering
            print(
                "Embedding confidence validation enabled: "
                f"min_similarity={embedding_min_similarity}, "
                f"min_margin={embedding_min_margin}, "
                f"max_assignment_gap={embedding_max_assignment_gap}"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and torch.cuda.is_available():
            print("CUDA available. Moving pyannote pipeline to GPU...")
            self.pipeline.to(torch.device("cuda"))
        else:
            print("Pyannote will run on CPU.")

    @staticmethod
    def _load_audio_as_waveform(audio_path):
        waveform, sample_rate = sf.read(audio_path, always_2d=True)
        waveform = np.asarray(waveform, dtype=np.float32)
        waveform = waveform.T
        return {"waveform": torch.from_numpy(waveform), "sample_rate": int(sample_rate)}

    @staticmethod
    def _merge_short_segments(segments, min_duration, verbose=False):
        """
        Clean up diarization segments shorter than `min_duration` seconds,
        which are usually clustering misfires rather than genuine turns.

        Two cases are handled differently:
        1. Sandwiched by the SAME speaker on both sides (A, short-B, A):
           strong signal the short segment is a misfire on an otherwise
           clean turn -- snap it to match its neighbors.
        2. No such signal: fall back to absorbing it into the segment
           immediately before it.

        Off by default (min_duration=0). This still throws away genuinely
        short interjections some of the time -- there's no way to
        perfectly distinguish "clustering noise" from "someone said one
        quick word" from timing alone. If a short reply keeps getting
        eaten, that's this heuristic being wrong for that specific spot,
        not proof anything else is broken -- lower min_duration or pass
        verbose=True to see exactly which segments were touched.
        """
        if min_duration <= 0 or len(segments) < 2:
            return segments

        result = [dict(s) for s in sorted(segments, key=lambda s: s["start"])]

        i = 1
        while i < len(result) - 1:
            seg = result[i]
            duration = seg["end"] - seg["start"]
            if duration < min_duration:
                prev_seg, next_seg = result[i - 1], result[i + 1]
                gap_prev = seg["start"] - prev_seg["end"]
                gap_next = next_seg["start"] - seg["end"]

                if prev_seg["speaker"] == next_seg["speaker"] and gap_prev < 1.0 and gap_next < 1.0:
                    if verbose:
                        print(f"  [merge] {seg['start']:.2f}-{seg['end']:.2f} "
                              f"({duration:.2f}s) relabeled {seg['speaker']} -> {prev_seg['speaker']} "
                              f"(sandwiched by matching neighbors)")
                    seg["speaker"] = prev_seg["speaker"]
                elif gap_prev < 1.0:
                    if verbose:
                        print(f"  [merge] {seg['start']:.2f}-{seg['end']:.2f} "
                              f"({duration:.2f}s, speaker {seg['speaker']}) absorbed into preceding "
                              f"{prev_seg['speaker']} segment")
                    prev_seg["end"] = max(prev_seg["end"], seg["end"])
                    del result[i]
                    continue
            i += 1

        # Edge segments (first/last) only have one neighbor to compare against.
        if len(result) >= 2:
            first = result[0]
            if (first["end"] - first["start"]) < min_duration and (result[1]["start"] - first["end"]) < 1.0:
                if verbose:
                    print(f"  [merge] leading {first['start']:.2f}-{first['end']:.2f} "
                          f"(speaker {first['speaker']}) absorbed into following {result[1]['speaker']}")
                result[1]["start"] = min(result[1]["start"], first["start"])
                result.pop(0)
        if len(result) >= 2:
            last = result[-1]
            if (last["end"] - last["start"]) < min_duration and (last["start"] - result[-2]["end"]) < 1.0:
                if verbose:
                    print(f"  [merge] trailing {last['start']:.2f}-{last['end']:.2f} "
                          f"(speaker {last['speaker']}) absorbed into preceding {result[-2]['speaker']}")
                result[-2]["end"] = max(result[-2]["end"], last["end"])
                result.pop()

        return result

    def run(self, audio_file, num_speakers=None, min_speakers=None, max_speakers=None,
            min_segment_duration=0.0, verbose=False):
        kwargs = {}
        if num_speakers is not None:
            print(f"Using exact number of speakers: {num_speakers}")
            kwargs["num_speakers"] = num_speakers
        else:
            print(f"Using speaker range: {min_speakers}-{max_speakers}")
            kwargs["min_speakers"] = min_speakers
            kwargs["max_speakers"] = max_speakers

        try:
            output = self.pipeline(audio_file, **kwargs)
        except RuntimeError as exc:
            if "torchcodec is not available" in str(exc) or "Cannot read audio file" in str(exc):
                print("TorchCodec unavailable; loading audio into memory as waveform dictionary...")
                audio = self._load_audio_as_waveform(audio_file)
                output = self.pipeline(audio, **kwargs)
            else:
                raise

        confidence_stats = (
            self.confidence_clustering.last_stats
            if self.confidence_clustering is not None
            else None
        )
        if confidence_stats is not None:
            print(
                "Embedding confidence validation: "
                f"checked {confidence_stats['checked']}, "
                f"rejected {confidence_stats['rejected']} "
                f"(low similarity {confidence_stats['low_similarity']}, "
                f"near tie {confidence_stats['ambiguous']}, "
                f"large constrained gap {confidence_stats['large_assignment_gap']})"
            )

        # ConfidenceAwareClustering appends a zero centroid for its rejection
        # cluster. Community-1 keeps centroid rows aligned with output labels,
        # so this reliably identifies the label that must be exposed as
        # UNKNOWN rather than as another real speaker.
        unknown_labels = set()
        output_embeddings = getattr(output, "speaker_embeddings", None)
        if output_embeddings is not None:
            output_labels = output.speaker_diarization.labels()
            for label, embedding in zip(output_labels, output_embeddings):
                if np.linalg.norm(embedding) <= EPSILON:
                    unknown_labels.add(label)

        # Community-1 provides exclusive speaker diarization, which is
        # useful when combining diarization with Whisper word timestamps.
        diarization = getattr(output, "exclusive_speaker_diarization", output.exclusive_speaker_diarization)

        segments = [] 
        for turn, speaker in diarization:
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": UNKNOWN_SPEAKER if speaker in unknown_labels else speaker,
            })

        segments.sort(key=lambda s: s["start"])

        # --- Diagnostics ---
        # If num_speakers was forced (e.g. 7) but one label shows up with
        # only a couple of tiny segments here, that's a strong sign the
        # true speaker count doesn't match the forced value, and the
        # clustering was forced to split or merge real speakers to comply.
        # Re-run with num_speakers=None and a min/max range instead.
        per_speaker = {}
        for s in segments:
            stats = per_speaker.setdefault(s["speaker"], {"count": 0, "duration": 0.0})
            stats["count"] += 1
            stats["duration"] += s["end"] - s["start"]

        real_speaker_count = sum(
            speaker != UNKNOWN_SPEAKER for speaker in per_speaker
        )
        print(f"Detected diarization segments: {len(segments)}")
        unknown_suffix = (
            " (+ UNKNOWN rejection track)"
            if UNKNOWN_SPEAKER in per_speaker
            else ""
        )
        print(f"Distinct speaker labels: {real_speaker_count}{unknown_suffix}")
        for spk, stats in sorted(per_speaker.items()):
            print(f"  {spk}: {stats['count']} segments, {stats['duration']:.1f}s total")

        if verbose:
            print("--- Raw segments (before any merging) ---")
            for s in segments:
                print(f"  {s['start']:.2f}-{s['end']:.2f} ({s['end']-s['start']:.2f}s): {s['speaker']}")

        if min_segment_duration > 0:
            before = len(segments)
            segments = self._merge_short_segments(segments, min_segment_duration, verbose=verbose)
            print(f"Merged short segments (<{min_segment_duration}s): {before} -> {len(segments)}")

        return segments
