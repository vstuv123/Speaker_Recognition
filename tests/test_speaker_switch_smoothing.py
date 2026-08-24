import unittest

from alignment.aligner import smooth_speaker_labels


def word(start, end, speaker, confidence=0.9, method="overlap"):
    item = {"start": start, "end": end, "word": " word", "speaker": speaker}
    if confidence is not None:
        item["speaker_confidence"] = confidence
    if method is not None:
        item["speaker_assignment_method"] = method
    return item


def smooth(words):
    return smooth_speaker_labels(
        words,
        min_duration=0.5,
        context_words=2,
        context_seconds=1.0,
        min_majority_ratio=0.5,
        protect_confidence=0.7,
    )


class SpeakerSwitchSmoothingTests(unittest.TestCase):
    def test_isolated_incorrect_speaker_is_corrected(self):
        words = [
            word(0.0, 0.2, "A"), word(0.2, 0.4, "A"),
            word(0.4, 0.55, "B", confidence=0.25),
            word(0.55, 0.8, "A"), word(0.8, 1.0, "A"),
        ]
        result = smooth(words)
        self.assertEqual([item["speaker"] for item in result], ["A"] * 5)
        self.assertEqual(result[2]["speaker_smoothed_from"], "B")

    def test_genuine_short_speaker_turn_is_protected_by_direct_evidence(self):
        words = [
            word(0.0, 0.3, "A"), word(0.3, 0.5, "A"),
            word(0.5, 0.7, "B", confidence=0.95),
            word(0.7, 0.9, "A"), word(0.9, 1.2, "A"),
        ]
        self.assertEqual([item["speaker"] for item in smooth(words)], ["A", "A", "B", "A", "A"])

    def test_normal_transition_is_preserved(self):
        words = [word(0.0, 0.3, "A"), word(0.3, 0.6, "A"), word(0.6, 0.9, "B")]
        self.assertEqual([item["speaker"] for item in smooth(words)], ["A", "A", "B"])

    def test_rapid_a_b_a_with_weak_evidence_is_corrected(self):
        words = [
            word(0.0, 0.2, "A"),
            word(0.2, 0.3, "B", confidence=0.2),
            word(0.3, 0.5, "A"),
        ]
        self.assertEqual([item["speaker"] for item in smooth(words)], ["A", "A", "A"])

    def test_multiple_speakers_do_not_form_false_context(self):
        words = [
            word(0.0, 0.2, "A"), word(0.2, 0.3, "B", confidence=0.2),
            word(0.3, 0.5, "C"), word(0.5, 0.7, "C"),
        ]
        self.assertEqual([item["speaker"] for item in smooth(words)], ["A", "B", "C", "C"])

    def test_missing_timestamp_is_left_unchanged(self):
        words = [word(0.0, 0.2, "A"), {"word": " x", "speaker": "B"}, word(0.3, 0.5, "A")]
        self.assertEqual([item["speaker"] for item in smooth(words)], ["A", "B", "A"])

    def test_missing_neighbor_timestamp_is_left_unchanged(self):
        words = [
            {"start": 0.0, "word": " a", "speaker": "A"},
            word(0.2, 0.3, "B", confidence=0.2),
            word(0.3, 0.5, "A"),
        ]
        self.assertEqual([item["speaker"] for item in smooth(words)], ["A", "B", "A"])

    def test_ambiguous_assignment_without_confidence_can_be_corrected(self):
        words = [
            word(0.0, 0.2, "A"),
            word(0.2, 0.3, "B", confidence=None, method="unresolved"),
            word(0.3, 0.5, "A"),
        ]
        self.assertEqual([item["speaker"] for item in smooth(words)], ["A", "A", "A"])


if __name__ == "__main__":
    unittest.main()
