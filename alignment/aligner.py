"""
Aligns Whisper word timestamps with pyannote diarization segments.
"""

from collections import defaultdict


# A match must cover either this much absolute time or this fraction of a
# word, whichever is smaller. The ratio branch keeps very short words viable.
MIN_MEANINGFUL_OVERLAP_SECONDS = 0.04
MIN_MEANINGFUL_OVERLAP_RATIO = 0.20
AMBIGUITY_MARGIN_RATIO = 0.15
CONTEXT_MAX_GAP_SECONDS = 1.0
EPSILON = 1e-9


def _overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _speaker_overlap_scores(start, end, diarization_segments):
    """
    Return total intersecting time per speaker.

    Intersections belonging to the same speaker are unioned before their
    durations are measured. This prevents duplicate/overlapping pyannote
    segments from making a speaker appear to overlap a word for longer than
    the word itself.
    """
    intervals = defaultdict(list)
    for segment in diarization_segments:
        intersection_start = max(start, segment["start"])
        intersection_end = min(end, segment["end"])
        if intersection_end - intersection_start > EPSILON:
            intervals[segment["speaker"]].append((intersection_start, intersection_end))

    scores = {}
    for speaker, speaker_intervals in intervals.items():
        speaker_intervals.sort()
        merged = []
        for interval_start, interval_end in speaker_intervals:
            if merged and interval_start <= merged[-1][1] + EPSILON:
                merged[-1][1] = max(merged[-1][1], interval_end)
            else:
                merged.append([interval_start, interval_end])
        scores[speaker] = sum(interval_end - interval_start for interval_start, interval_end in merged)
    return scores


def _alignment_evidence(start, end, diarization_segments):
    word_duration = max(end - start, EPSILON)
    scores = _speaker_overlap_scores(start, end, diarization_segments)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))

    if not ranked:
        return {
            "speaker": "UNKNOWN", "confidence": 0.0, "margin": 0.0,
            "meaningful": False, "ambiguous": False, "scores": {},
        }

    best_speaker, best_overlap = ranked[0]
    second_overlap = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = min(1.0, best_overlap / word_duration)
    margin = (best_overlap - second_overlap) / word_duration
    required_overlap = min(
        MIN_MEANINGFUL_OVERLAP_SECONDS,
        word_duration * MIN_MEANINGFUL_OVERLAP_RATIO,
    )
    meaningful = best_overlap + EPSILON >= required_overlap
    ambiguous = len(ranked) > 1 and margin < AMBIGUITY_MARGIN_RATIO
    return {
        "speaker": best_speaker,
        "confidence": confidence,
        "margin": margin,
        "meaningful": meaningful,
        "ambiguous": ambiguous,
        "scores": dict(ranked),
    }


def find_speaker(start, end, diarization_segments):
    """Return the strongest overlap speaker and overlap/word confidence."""
    evidence = _alignment_evidence(start, end, diarization_segments)
    if not evidence["meaningful"]:
        return "UNKNOWN", evidence["confidence"]
    return evidence["speaker"], evidence["confidence"]


def _context_speaker(index, word_items, provisional, evidence):
    """Resolve an uncertain word only when nearby reliable context agrees."""
    word = word_items[index]
    previous = None
    following = None

    for other_index in range(index - 1, -1, -1):
        if provisional[other_index] != "UNKNOWN":
            gap = max(0.0, word["start"] - word_items[other_index]["end"])
            if gap <= CONTEXT_MAX_GAP_SECONDS:
                previous = provisional[other_index]
            break

    for other_index in range(index + 1, len(word_items)):
        if provisional[other_index] != "UNKNOWN":
            gap = max(0.0, word_items[other_index]["start"] - word["end"])
            if gap <= CONTEXT_MAX_GAP_SECONDS:
                following = provisional[other_index]
            break

    if previous and following:
        return previous if previous == following else "UNKNOWN"
    if previous or following:
        return previous or following

    # With no usable context, retain a unique weak-overlap winner, but never
    # resolve a near-tie according to segment input order.
    if evidence["meaningful"] and not evidence["ambiguous"]:
        return evidence["speaker"]
    return "UNKNOWN"


def assign_speakers_to_words(word_items, diarization_segments):
    print("Assigning speaker to each word...")

    evidence_items = [
        _alignment_evidence(word["start"], word["end"], diarization_segments)
        for word in word_items
    ]
    provisional = [
        evidence["speaker"]
        if evidence["meaningful"] and not evidence["ambiguous"]
        else "UNKNOWN"
        for evidence in evidence_items
    ]

    context_resolved = 0
    unresolved = 0
    for index, (word, evidence) in enumerate(zip(word_items, evidence_items)):
        speaker = provisional[index]
        method = "overlap"
        # UNKNOWN can now be explicit diarization evidence from the embedding
        # confidence validator. Do not overwrite that deliberate rejection
        # with neighboring context; context is only for gaps or overlap ties.
        explicit_embedding_unknown = (
            evidence["meaningful"]
            and not evidence["ambiguous"]
            and evidence["speaker"] == "UNKNOWN"
        )
        if explicit_embedding_unknown:
            unresolved += 1
            method = "embedding_uncertain"
        elif speaker == "UNKNOWN":
            speaker = _context_speaker(index, word_items, provisional, evidence)
            if speaker != "UNKNOWN":
                context_resolved += 1
                method = "context"
            else:
                unresolved += 1
                method = "unresolved"

        word["speaker"] = speaker
        word["speaker_confidence"] = evidence["confidence"]
        word["speaker_overlap_scores"] = evidence["scores"]
        word["speaker_assignment_method"] = method

    ambiguous = sum(item["ambiguous"] for item in evidence_items)
    no_overlap = sum(not item["scores"] for item in evidence_items)
    print(
        "Alignment summary: "
        f"{len(word_items) - context_resolved - unresolved} direct overlap, "
        f"{context_resolved} context-resolved, {unresolved} unresolved; "
        f"{ambiguous} ambiguous, {no_overlap} with zero overlap"
    )
    for word in [w for w in word_items if w["speaker_assignment_method"] == "unresolved"][:5]:
        print(
            f"  Unresolved word {word['start']:.3f}-{word['end']:.3f} "
            f"{word['word']!r}; overlap scores={word['speaker_overlap_scores']}"
        )

    return word_items


def smooth_speaker_labels(
    word_items,
    window_seconds=None,
    min_majority_ratio=0.66,
    protect_confidence=0.7,
    *,
    min_duration=0.6,
    context_words=2,
    context_seconds=None,
):
    """
    Correct short, weakly-supported speaker islands in an A -> B -> A pattern.

    A complete contiguous speaker run is considered, rather than changing
    individual words from a general majority vote. A run is relabeled only
    when it is short, both adjacent runs have the same speaker, nearby words
    on both sides support that speaker, and the run has no strong direct
    overlap assignment. Ordinary A -> B transitions and strongly-supported
    short replies are therefore preserved.

    ``window_seconds`` remains accepted as the legacy name for
    ``context_seconds``. Existing callers using the old interface continue
    to work.
    """
    if len(word_items) < 3 or context_words < 1 or min_duration <= 0:
        return word_items

    if context_seconds is None:
        context_seconds = window_seconds if window_seconds is not None else 1.5

    # Build runs from an immutable label snapshot so one correction cannot
    # influence the eligibility of a later candidate in the same pass.
    labels = [word.get("speaker", "UNKNOWN") for word in word_items]
    runs = []
    run_start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[run_start]:
            runs.append({"start": run_start, "end": index - 1, "speaker": labels[run_start]})
            run_start = index

    changed_words = 0
    changed_runs = 0
    protected_runs = 0
    skipped_missing_time = 0

    for run_index in range(1, len(runs) - 1):
        run = runs[run_index]
        previous_run = runs[run_index - 1]
        next_run = runs[run_index + 1]
        replacement = previous_run["speaker"]

        # Only an isolated speaker island is eligible. This deliberately
        # leaves A -> B, A -> B -> C, and UNKNOWN islands untouched.
        if (
            replacement != next_run["speaker"]
            or run["speaker"] in (replacement, "UNKNOWN")
        ):
            continue

        candidate_words = word_items[run["start"]:run["end"] + 1]
        if any("start" not in word or "end" not in word for word in candidate_words):
            skipped_missing_time += 1
            continue
        start = candidate_words[0]["start"]
        end = candidate_words[-1]["end"]
        if start is None or end is None or end < start:
            skipped_missing_time += 1
            continue
        if end - start > min_duration + EPSILON:
            continue

        left_end = word_items[run["start"] - 1].get("end")
        right_start = word_items[run["end"] + 1].get("start")
        if left_end is None or right_start is None:
            skipped_missing_time += 1
            continue
        left_gap = max(0.0, start - left_end)
        right_gap = max(0.0, right_start - end)
        if left_gap > context_seconds or right_gap > context_seconds:
            continue

        left_context = word_items[max(0, run["start"] - context_words):run["start"]]
        right_context = word_items[
            run["end"] + 1:min(len(word_items), run["end"] + 1 + context_words)
        ]
        if not left_context or not right_context:
            continue
        left_support = sum(word.get("speaker") == replacement for word in left_context)
        right_support = sum(word.get("speaker") == replacement for word in right_context)
        if (
            left_support / len(left_context) < min_majority_ratio
            or right_support / len(right_context) < min_majority_ratio
        ):
            continue

        # A high-confidence direct overlap is evidence for a real short turn.
        # Missing confidence is not treated as strong evidence: older/custom
        # word records may not contain that diagnostic field.
        has_strong_direct_evidence = any(
            word.get("speaker_assignment_method", "overlap") == "overlap"
            and word.get("speaker_confidence", 0.0) >= protect_confidence
            for word in candidate_words
        )
        if has_strong_direct_evidence:
            protected_runs += 1
            continue

        for word in candidate_words:
            word["speaker_smoothed_from"] = word["speaker"]
            word["speaker"] = replacement
            word["speaker_assignment_method"] = "speaker_switch_smoothing"
            changed_words += 1
        changed_runs += 1

    print(
        f"Speaker-switch smoothing: corrected {changed_runs} short runs "
        f"({changed_words} words); protected {protected_runs} strong short runs; "
        f"skipped {skipped_missing_time} runs with missing timestamps"
    )
    return word_items
