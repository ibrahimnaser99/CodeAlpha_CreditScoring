import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.model_selection import ParameterSampler

from preprocessing import get_feature_columns, split_xy
from tests.test_modeling import make_dataset
from tuning import (
    SEARCH_N_JOBS,
    SEARCH_SPACES,
    apply_best_params,
    build_candidates,
    load_best_params,
    save_best_params,
    to_builtin,
)


class TestCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_dataset(n=400)
        cls.features = get_feature_columns(cls.df)
        cls.X, cls.y = split_xy(cls.df, cls.features)

    def test_every_candidate_has_a_search_space_and_n_jobs(self):
        names = set(build_candidates(self.features))

        self.assertEqual(names, set(SEARCH_SPACES))
        self.assertEqual(names, set(SEARCH_N_JOBS))

    def test_sampled_parameters_are_valid_and_the_model_fits(self):
        for name, pipeline in build_candidates(self.features).items():
            with self.subTest(model=name):
                sampled = next(
                    iter(ParameterSampler(SEARCH_SPACES[name], n_iter=1, random_state=0))
                )

                model = apply_best_params(pipeline, sampled).fit(self.X, self.y)
                proba = model.predict_proba(self.X)

                self.assertEqual(proba.shape, (len(self.X), 2))
                self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_apply_best_params_does_not_change_the_original(self):
        pipeline = build_candidates(self.features)["Decision Tree"]
        before = pipeline.get_params()["model__max_depth"]

        tuned = apply_best_params(pipeline, {"model__max_depth": 3})

        self.assertEqual(tuned.get_params()["model__max_depth"], 3)
        self.assertEqual(pipeline.get_params()["model__max_depth"], before)


class TestSavingParams(unittest.TestCase):
    def test_numpy_numbers_become_plain_numbers(self):
        self.assertIs(type(to_builtin(np.int64(3))), int)
        self.assertIs(type(to_builtin(np.float64(0.5))), float)
        self.assertIsNone(to_builtin(None))
        self.assertEqual(to_builtin("sqrt"), "sqrt")

    def test_params_survive_a_save_and_load(self):
        params = {
            "Random Forest": {
                "model__max_depth": None,
                "model__min_samples_leaf": np.int64(12),
                "model__max_features": "sqrt",
            },
            "Logistic Regression": {"model__C": np.float64(0.25)},
        }

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "best_params.json"
            save_best_params(path, params)
            json.loads(path.read_text(encoding="utf-8"))
            loaded = load_best_params(path)

        self.assertEqual(loaded["Random Forest"]["model__min_samples_leaf"], 12)
        self.assertIsNone(loaded["Random Forest"]["model__max_depth"])
        self.assertEqual(loaded["Logistic Regression"]["model__C"], 0.25)


if __name__ == "__main__":
    unittest.main()
