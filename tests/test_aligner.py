import unittest

from alignment.aligner import _overlap, _speaker_overlap_scores, assign_speakers_to_words


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def word(start, end, text=" word"):
    return {"start": start, "end": end, "word": text}


class TemporalOverlapTests(unittest.TestCase):
    def test_explicit_embedding_unknown_is_not_overwritten_by_context(self):
        words = [
            {"start": 0.0, "end": 0.4, "word": " A"},
            {"start": 0.4, "end": 0.8, "word": " uncertain"},
            {"start": 0.8, "end": 1.2, "word": " A"},
        ]
        diarization = [
            {"start": 0.0, "end": 0.4, "speaker": "A"},
            {"start": 0.4, "end": 0.8, "speaker": "UNKNOWN"},
            {"start": 0.8, "end": 1.2, "speaker": "A"},
        ]

        result = assign_speakers_to_words(words, diarization)

        self.assertEqual(result[1]["speaker"], "UNKNOWN")
        self.assertEqual(result[1]["speaker_assignment_method"], "embedding_uncertain")

    def test_overlap_calculation(self):
        self.assertEqual(_overlap(1.0, 3.0, 2.0, 4.0), 1.0)
        self.assertEqual(_overlap(1.0, 2.0, 2.0, 3.0), 0)

    def test_word_fully_contained(self):
        result = assign_speakers_to_words([word(1.0, 1.5)], [turn(0.0, 2.0, "A")])
        self.assertEqual(result[0]["speaker"], "A")
        self.assertEqual(result[0]["speaker_confidence"], 1.0)

    def test_boundary_word_uses_stronger_overlap(self):
        result = assign_speakers_to_words(
            [word(1.8, 2.5)],
            [turn(0.0, 2.0, "A"), turn(2.0, 4.0, "B")],
        )
        self.assertEqual(result[0]["speaker"], "B")

    def test_equal_boundary_split_remains_unknown_when_context_disagrees(self):
        words = [word(1.0, 1.5), word(1.8, 2.2), word(2.5, 3.0)]
        turns = [turn(0.0, 2.0, "A"), turn(2.0, 4.0, "B")]
        result = assign_speakers_to_words(words, turns)
        self.assertEqual([item["speaker"] for item in result], ["A", "UNKNOWN", "B"])

    def test_gap_word_uses_agreeing_context(self):
        words = [word(0.5, 0.8), word(1.05, 1.15), word(1.3, 1.6)]
        turns = [turn(0.0, 1.0, "A"), turn(1.2, 2.0, "A")]
        result = assign_speakers_to_words(words, turns)
        self.assertEqual(result[1]["speaker"], "A")
        self.assertEqual(result[1]["speaker_assignment_method"], "context")

    def test_isolated_gap_word_stays_unknown(self):
        result = assign_speakers_to_words([word(5.0, 5.2)], [turn(0.0, 1.0, "A")])
        self.assertEqual(result[0]["speaker"], "UNKNOWN")

    def test_very_short_word_can_match(self):
        result = assign_speakers_to_words([word(1.000, 1.010)], [turn(1.0, 1.01, "A")])
        self.assertEqual(result[0]["speaker"], "A")

    def test_multiple_segments_for_same_speaker_are_unioned(self):
        scores = _speaker_overlap_scores(
            0.0, 1.0,
            [turn(0.0, 0.8, "A"), turn(0.2, 1.0, "A"), turn(0.4, 0.6, "B")],
        )
        self.assertAlmostEqual(scores["A"], 1.0)
        self.assertAlmostEqual(scores["B"], 0.2)

    def test_multiple_speakers_uses_strongest_total_overlap(self):
        result = assign_speakers_to_words(
            [word(0.0, 1.0)],
            [turn(0.0, 0.8, "A"), turn(0.5, 1.0, "B")],
        )
        self.assertEqual(result[0]["speaker"], "A")


if __name__ == "__main__":
    unittest.main()
