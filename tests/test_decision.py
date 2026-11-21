import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from evaluation import (
    DEFAULT_THRESHOLD_GRID,
    bootstrap_ci,
    brier,
    calibration_table,
    cost_optimal_threshold,
    cost_per_client,
)
from final_model import (
    build_final_estimator,
    ensure_test_not_used,
    load_json,
    load_model,
    save_json,
    save_model,
)
from preprocessing import get_feature_columns, split_xy
from tests.test_modeling import make_dataset


def make_scores(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    proba = np.clip(y * 0.3 + rng.random(n) * 0.7, 0, 1)
    return y, proba


class TestCost(unittest.TestCase):
    def test_cost_per_client_known_values(self):
        y = [0, 0, 1, 1]
        proba = [0.1, 0.6, 0.4, 0.9]

        cost = cost_per_client(y, proba, 0.5, cost_fn=3, cost_fp=1)

        self.assertAlmostEqual(cost, (3 * 1 + 1 * 1) / 4)

    def test_perfect_model_costs_nothing(self):
        threshold, cost = cost_optimal_threshold(
            [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], cost_fn=3, cost_fp=1
        )

        self.assertEqual(cost, 0.0)
        self.assertTrue(0.2 < threshold <= 0.8)

    def test_expensive_missed_defaulters_lower_the_threshold(self):
        y, proba = make_scores()

        threshold_missed_is_costly, _ = cost_optimal_threshold(y, proba, cost_fn=10, cost_fp=1)
        threshold_rejection_is_costly, _ = cost_optimal_threshold(y, proba, cost_fn=1, cost_fp=10)

        self.assertLess(threshold_missed_is_costly, threshold_rejection_is_costly)

    def test_optimal_threshold_matches_a_slow_loop(self):
        y, proba = make_scores(n=600, seed=3)

        _, fast_cost = cost_optimal_threshold(y, proba, cost_fn=3, cost_fp=1)
        slow_cost = min(
            cost_per_client(y, proba, threshold, 3, 1)
            for threshold in DEFAULT_THRESHOLD_GRID
        )

        self.assertAlmostEqual(fast_cost, slow_cost)


class TestCalibration(unittest.TestCase):
    def test_group_sizes_add_up(self):
        y, proba = make_scores()

        table = calibration_table(y, proba, n_bins=10)

        self.assertEqual(table["clients"].sum(), len(y))

    def test_well_calibrated_probabilities_match_observed_rates(self):
        rng = np.random.default_rng(1)
        proba = rng.random(30_000)
        y = (rng.random(30_000) < proba).astype(int)

        table = calibration_table(y, proba, n_bins=10)

        self.assertLess(
            (table["mean_predicted"] - table["observed_rate"]).abs().max(), 0.03
        )

    def test_constant_scores_give_one_group(self):
        table = calibration_table([0, 1, 0, 1], [0.3, 0.3, 0.3, 0.3])

        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(table.loc[0, "observed_rate"], 0.5)

    def test_brier_known_values(self):
        self.assertAlmostEqual(brier([0, 1], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(brier([0, 1], [0.5, 0.5]), 0.25)


class TestBootstrap(unittest.TestCase):
    def test_interval_is_ordered_and_contains_the_estimate(self):
        y, proba = make_scores()

        low, high = bootstrap_ci(y, proba, roc_auc_score, n_resamples=300)

        self.assertLess(low, high)
        self.assertTrue(low <= roc_auc_score(y, proba) <= high)

    def test_same_seed_gives_same_interval(self):
        y, proba = make_scores()

        first = bootstrap_ci(y, proba, roc_auc_score, n_resamples=200, seed=5)
        second = bootstrap_ci(y, proba, roc_auc_score, n_resamples=200, seed=5)

        self.assertEqual(first, second)

    def test_rare_class_does_not_crash(self):
        y = np.array([0] * 60 + [1])
        proba = np.linspace(0, 1, 61)

        low, high = bootstrap_ci(y, proba, roc_auc_score, n_resamples=200)

        self.assertLessEqual(low, high)


class TestFinalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_dataset(n=400)
        cls.features = get_feature_columns(cls.df)
        cls.X, cls.y = split_xy(cls.df, cls.features)
        cls.params = {"Decision Tree": {"model__max_depth": 3}}

    def test_calibrate_flag_chooses_the_wrapper(self):
        plain = build_final_estimator(
            self.features, "Decision Tree", self.params, calibrate=False
        )
        wrapped = build_final_estimator(
            self.features, "Decision Tree", self.params, calibrate=True
        )

        self.assertIsInstance(plain, Pipeline)
        self.assertIsInstance(wrapped, CalibratedClassifierCV)

    def test_tuned_parameters_are_applied(self):
        plain = build_final_estimator(
            self.features, "Decision Tree", self.params, calibrate=False
        )

        self.assertEqual(plain.get_params()["model__max_depth"], 3)

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_final_estimator(self.features, "Magic Model", {})

    def test_calibrated_model_gives_probabilities(self):
        model = build_final_estimator(
            self.features, "Decision Tree", self.params, calibrate=True
        )

        proba = model.fit(self.X, self.y).predict_proba(self.X)

        self.assertEqual(proba.shape, (len(self.X), 2))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_saved_model_gives_the_same_predictions(self):
        model = build_final_estimator(
            self.features, "Decision Tree", self.params, calibrate=False
        ).fit(self.X, self.y)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "models" / "final.joblib"
            save_model(path, model, self.features)
            loaded, loaded_features = load_model(path)

        self.assertEqual(loaded_features, self.features)
        np.testing.assert_allclose(loaded.predict_proba(self.X), model.predict_proba(self.X))

    def test_missing_files_give_a_clear_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_model(Path(folder) / "missing.joblib")
            with self.assertRaises(FileNotFoundError):
                load_json(Path(folder) / "missing.json")

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reports" / "config.json"
            save_json(path, {"threshold": 0.25, "model": "Random Forest"})

            self.assertEqual(load_json(path), {"threshold": 0.25, "model": "Random Forest"})


class TestTestSetGuard(unittest.TestCase):
    def test_first_run_is_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            ensure_test_not_used(Path(folder) / "final_test_results.json")

    def test_second_run_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "final_test_results.json"
            path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "already exists"):
                ensure_test_not_used(path)

    def test_force_allows_a_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "final_test_results.json"
            path.write_text("{}", encoding="utf-8")

            ensure_test_not_used(path, force=True)


if __name__ == "__main__":
    unittest.main()
