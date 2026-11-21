import unittest

import numpy as np

from evaluation import best_f1_threshold, ranking_metrics, threshold_metrics


class TestThresholdMetrics(unittest.TestCase):
    def test_known_confusion_matrix(self):
        y_true = [0, 0, 1, 1]
        proba = [0.1, 0.6, 0.4, 0.9]

        result = threshold_metrics(y_true, proba, 0.5)

        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_never_predicting_default_gives_zero_not_an_error(self):
        result = threshold_metrics([0, 0, 1, 1], [0.2, 0.2, 0.2, 0.2], 0.5)

        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)
        self.assertEqual((result["tp"], result["fp"]), (0, 0))

    def test_threshold_is_inclusive(self):
        result = threshold_metrics([1], [0.5], 0.5)

        self.assertEqual(result["tp"], 1)

    def test_lower_threshold_never_lowers_recall(self):
        rng = np.random.default_rng(0)
        y_true = (rng.random(500) < 0.3).astype(int)
        proba = np.clip(y_true * 0.3 + rng.random(500) * 0.7, 0, 1)

        strict = threshold_metrics(y_true, proba, 0.7)["recall"]
        lenient = threshold_metrics(y_true, proba, 0.3)["recall"]

        self.assertGreaterEqual(lenient, strict)


class TestBestF1Threshold(unittest.TestCase):
    def test_perfectly_separable_scores(self):
        threshold, f1 = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

        self.assertAlmostEqual(f1, 1.0)
        self.assertAlmostEqual(threshold, 0.8)

    def test_chosen_threshold_reproduces_the_reported_f1(self):
        rng = np.random.default_rng(1)
        y_true = (rng.random(400) < 0.25).astype(int)
        proba = np.clip(y_true * 0.25 + rng.random(400) * 0.75, 0, 1)

        threshold, f1 = best_f1_threshold(y_true, proba)

        self.assertAlmostEqual(threshold_metrics(y_true, proba, threshold)["f1"], f1)


class TestRankingMetrics(unittest.TestCase):
    def test_perfect_ranking(self):
        result = ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

        self.assertAlmostEqual(result["roc_auc"], 1.0)
        self.assertAlmostEqual(result["pr_auc"], 1.0)

    def test_constant_scores_have_no_skill(self):
        y_true = [0, 0, 0, 1]

        result = ranking_metrics(y_true, [0.3, 0.3, 0.3, 0.3])

        self.assertAlmostEqual(result["roc_auc"], 0.5)
        self.assertAlmostEqual(result["pr_auc"], 0.25)


if __name__ == "__main__":
    unittest.main()
