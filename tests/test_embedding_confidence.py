import unittest

import numpy as np

from diarization.diarizer import validate_embedding_assignments


def validate(embedding, assigned, centroids, segmentation=None, **overrides):
    embeddings = np.asarray(embedding, dtype=float).reshape(1, 1, -1)
    hard = np.asarray([[assigned]], dtype=int)
    if segmentation is None:
        segmentation = np.ones((1, 4, 1), dtype=float)
    settings = {
        "min_similarity": 0.35,
        "min_margin": 0.05,
        "max_assignment_gap": 0.20,
    }
    settings.update(overrides)
    return validate_embedding_assignments(
        embeddings,
        hard,
        np.asarray(centroids, dtype=float),
        segmentation,
        **settings,
    )


class EmbeddingConfidenceTests(unittest.TestCase):
    def test_strong_assignment_is_preserved(self):
        hard, rejected, stats = validate([1.0, 0.0], 0, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(hard[0, 0], 0)
        self.assertFalse(rejected[0, 0])
        self.assertEqual(stats["rejected"], 0)

    def test_low_similarity_is_sent_to_unknown_cluster(self):
        hard, rejected, stats = validate([-1.0, -1.0], 0, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(hard[0, 0], 2)
        self.assertTrue(rejected[0, 0])
        self.assertEqual(stats["low_similarity"], 1)

    def test_near_tie_is_sent_to_unknown_cluster(self):
        hard, rejected, stats = validate([1.0, 1.0], 0, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(hard[0, 0], 2)
        self.assertTrue(rejected[0, 0])
        self.assertEqual(stats["ambiguous"], 1)

    def test_small_constrained_assignment_gap_is_preserved(self):
        hard, rejected, _ = validate(
            [1.0, 0.0],
            1,
            [[1.0, 0.0], [0.98, 0.20]],
        )
        self.assertEqual(hard[0, 0], 1)
        self.assertFalse(rejected[0, 0])

    def test_large_constrained_assignment_gap_is_rejected(self):
        hard, rejected, stats = validate(
            [1.0, 0.0],
            1,
            [[1.0, 0.0], [0.5, 0.866]],
        )
        self.assertEqual(hard[0, 0], 2)
        self.assertTrue(rejected[0, 0])
        self.assertEqual(stats["large_assignment_gap"], 1)

    def test_short_but_strong_active_track_is_not_deleted(self):
        # Only one of four local frames is active. Confidence, not duration,
        # controls the decision.
        segmentation = np.asarray([[[1.0], [0.0], [0.0], [0.0]]])
        hard, rejected, _ = validate(
            [1.0, 0.0], 0, [[1.0, 0.0], [0.0, 1.0]], segmentation
        )
        self.assertEqual(hard[0, 0], 0)
        self.assertFalse(rejected[0, 0])

    def test_inactive_track_is_ignored(self):
        segmentation = np.zeros((1, 4, 1), dtype=float)
        hard, rejected, stats = validate(
            [-1.0, -1.0], 0, [[1.0, 0.0], [0.0, 1.0]], segmentation
        )
        self.assertEqual(hard[0, 0], 0)
        self.assertFalse(rejected[0, 0])
        self.assertEqual(stats["checked"], 0)


if __name__ == "__main__":
    unittest.main()
