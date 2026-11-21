import unittest

import numpy as np
import pandas as pd

from credit_data import EXPECTED_COLUMNS, TARGET
from modeling import build_models
from preprocessing import add_features, get_feature_columns, split_xy


def make_dataset(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Create a small random dataset with the columns used by the real data."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({column: rng.integers(0, 50_000, n) for column in EXPECTED_COLUMNS})
    df["limit_bal"] = rng.integers(10_000, 500_000, n)
    df["education"] = rng.integers(1, 5, n)
    df["marriage"] = rng.integers(1, 4, n)
    for month in range(1, 7):
        df[f"pay_{month}"] = rng.integers(-2, 4, n)
    df[TARGET] = (rng.random(n) < (0.15 + 0.3 * (df["pay_1"] > 0))).astype(int)
    return add_features(df)


class TestBuildModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_dataset()
        cls.features = get_feature_columns(cls.df)
        cls.X, cls.y = split_xy(cls.df, cls.features)

    def test_expected_models_exist(self):
        models = build_models(self.features)

        self.assertEqual(
            list(models), ["Dummy (prior)", "Logistic Regression", "Decision Tree", "Random Forest"]
        )

    def test_every_model_gives_probabilities(self):
        for name, model in build_models(self.features).items():
            with self.subTest(model=name):
                model.fit(self.X, self.y)
                proba = model.predict_proba(self.X)

                self.assertEqual(proba.shape, (len(self.X), 2))
                self.assertTrue(((proba >= 0) & (proba <= 1)).all())
                np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_dummy_predicts_the_default_rate(self):
        model = build_models(self.features)["Dummy (prior)"].fit(self.X, self.y)

        proba = model.predict_proba(self.X)[:, 1]

        np.testing.assert_allclose(proba, self.y.mean())

    def test_unseen_category_does_not_crash(self):
        X_new = self.X.head(5).copy()
        X_new["education"] = 9
        model = build_models(self.features)["Logistic Regression"].fit(self.X, self.y)

        proba = model.predict_proba(X_new)

        self.assertEqual(proba.shape, (5, 2))

    def test_scaler_is_fitted_on_training_data_only(self):
        model = build_models(self.features)["Logistic Regression"].fit(self.X, self.y)
        scaler = model.named_steps["prep"].named_transformers_["numeric"]

        self.assertAlmostEqual(
            scaler.mean_[list(scaler.feature_names_in_).index("limit_bal")],
            self.X["limit_bal"].mean(),
        )

    def test_sex_and_target_are_not_used(self):
        self.assertNotIn("sex", self.features)
        self.assertNotIn(TARGET, self.features)


if __name__ == "__main__":
    unittest.main()
