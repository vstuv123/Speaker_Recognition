import unittest

from diarization_evaluation.evaluate import (
    build_experiment_configurations,
    detect_overlap_regions,
    diarization_diagnostics,
    rank_results,
    word_diagnostics,
)


def segment(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


class EvaluationMetricsTests(unittest.TestCase):
    def test_experiment_one_values(self):
        configs = build_experiment_configurations()
        self.assertEqual([item["threshold"] for item in configs], [0.55, 0.60, 0.65, 0.70])
        self.assertTrue(all(item["speaker_mode"] == "automatic" for item in configs))

    def test_overlap_detection_supports_unique_speakers(self):
        regions = detect_overlap_regions([
            segment(0, 2, "A"), segment(1, 3, "B"), segment(1.5, 2.5, "C")
        ])
        self.assertEqual(regions[0]["speakers"], ["A", "B"])
        self.assertEqual(regions[1]["speakers"], ["A", "B", "C"])

    def test_diarization_metrics(self):
        metrics = diarization_diagnostics(
            [segment(0, 1, "A"), segment(1.2, 1.3, "B")],
            [segment(0, 1, "A"), segment(1.2, 1.3, "B")],
            2.0,
        )
        self.assertEqual(metrics["detected_speakers"], 2)
        self.assertEqual(metrics["speaker_turns"], 2)
        self.assertEqual(metrics["extremely_short_turns"], 1)
        self.assertEqual(metrics["gap_count"], 1)

    def test_word_metrics_detect_switches_and_aba(self):
        words = [
            {"start": 0, "end": 0.2, "speaker": "A"},
            {"start": 0.2, "end": 0.3, "speaker": "B"},
            {"start": 0.3, "end": 0.5, "speaker": "A"},
        ]
        metrics = word_diagnostics(words)
        self.assertEqual(metrics["word_switches"], 2)
        self.assertEqual(metrics["isolated_one_word_assignments"], 1)
        self.assertEqual(metrics["suspicious_aba_patterns"], 1)

    def test_ranking_prefers_stable_result(self):
        base = {
            "speaker_turns": 10, "word_count": 100, "detected_speakers": 3,
            "total_speech_duration": 60, "speaker_duration_balance_cv": 0.2,
            "minor_speaker_count": 0, "short_overlap_regions": 0,
            "unknown_word_percentage": 0,
        }
        stable = dict(base, configuration_id="stable", fragmentation_per_speech_minute=10,
                      extremely_short_turns=1, word_switches=5, rapid_word_switches=2,
                      isolated_one_word_assignments=0, suspicious_aba_patterns=0)
        noisy = dict(base, configuration_id="noisy", fragmentation_per_speech_minute=30,
                     extremely_short_turns=7, word_switches=30, rapid_word_switches=20,
                     isolated_one_word_assignments=8, suspicious_aba_patterns=8)
        ranked = rank_results([noisy, stable])
        self.assertEqual(ranked[0]["configuration_id"], "stable")
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])


if __name__ == "__main__":
    unittest.main()
