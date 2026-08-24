import unittest

from config import Config
from main import apply_speaker_count_args, parse_args


class SpeakerCountConfigurationTests(unittest.TestCase):
    def test_unknown_recording_defaults_to_automatic_range(self):
        cfg = Config()
        self.assertIsNone(cfg.num_speakers)
        self.assertEqual(cfg.min_speakers, 2)
        self.assertEqual(cfg.max_speakers, 10)
        self.assertEqual(cfg.diarization_clustering_threshold, 0.55)
        self.assertEqual(cfg.diarization_clustering_fb, 0.50)
        self.assertEqual(cfg.diarization_segmentation_min_duration_off, 0.20)
        self.assertTrue(cfg.diarization_embedding_validation)
        self.assertEqual(cfg.diarization_embedding_min_similarity, 0.35)
        self.assertEqual(cfg.diarization_embedding_min_margin, 0.05)
        self.assertEqual(cfg.diarization_embedding_max_assignment_gap, 0.20)

        args = parse_args(cfg, [])
        apply_speaker_count_args(cfg, args)
        self.assertIsNone(cfg.num_speakers)
        self.assertEqual((cfg.min_speakers, cfg.max_speakers), (2, 10))

    def test_custom_automatic_range(self):
        cfg = Config()
        args = parse_args(cfg, [
            "--auto-speakers", "--min-speakers", "3", "--max-speakers", "6"
        ])
        apply_speaker_count_args(cfg, args)
        self.assertIsNone(cfg.num_speakers)
        self.assertEqual((cfg.min_speakers, cfg.max_speakers), (3, 6))

    def test_explicit_count_takes_precedence(self):
        cfg = Config()
        args = parse_args(cfg, [
            "--num-speakers", "4", "--min-speakers", "2", "--max-speakers", "7"
        ])
        apply_speaker_count_args(cfg, args)
        self.assertEqual(cfg.num_speakers, 4)

    def test_explicit_count_even_takes_precedence_over_legacy_auto_flag(self):
        cfg = Config()
        args = parse_args(cfg, ["--auto-speakers", "--num-speakers", "5"])
        apply_speaker_count_args(cfg, args)
        self.assertEqual(cfg.num_speakers, 5)

    def test_equals_form_is_also_an_explicit_count(self):
        cfg = Config()
        args = parse_args(cfg, ["--auto-speakers", "--num-speakers=3"])
        apply_speaker_count_args(cfg, args)
        self.assertEqual(cfg.num_speakers, 3)

    def test_legacy_auto_flag_can_override_configured_exact_default(self):
        cfg = Config(num_speakers=6)
        args = parse_args(cfg, ["--auto-speakers"])
        apply_speaker_count_args(cfg, args)
        self.assertIsNone(cfg.num_speakers)

    def test_invalid_range_is_rejected(self):
        cfg = Config()
        args = parse_args(cfg, ["--min-speakers", "7", "--max-speakers", "2"])
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            apply_speaker_count_args(cfg, args)


if __name__ == "__main__":
    unittest.main()
