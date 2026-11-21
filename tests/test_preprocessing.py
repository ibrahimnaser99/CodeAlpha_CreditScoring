import unittest

import numpy as np
import pandas as pd

from credit_data import EXPECTED_COLUMNS, TARGET
from preprocessing import (
    ENGINEERED_FEATURES,
    add_features,
    get_feature_columns,
    split_data,
    split_xy,
)


def make_client(**overrides) -> dict:
    row = {column: 0 for column in EXPECTED_COLUMNS}
    row.update({"limit_bal": 100_000, "education": 2, "marriage": 1})
    row.update(overrides)
    return row


class TestAddFeatures(unittest.TestCase):
    def test_values_for_a_known_client(self):
        client = make_client(
            bill_1=50_000, bill_2=0, bill_3=-1_000, bill_4=20_000, bill_5=0, bill_6=0,
            pay_amt_1=10_000, pay_amt_4=5_000,
            pay_1=2, pay_2=0, pay_3=-1, pay_4=-2, pay_5=0, pay_6=1,
        )

        result = add_features(pd.DataFrame([client])).iloc[0]

        self.assertAlmostEqual(result["avg_utilization"], 0.7 / 6)
        self.assertAlmostEqual(result["max_utilization"], 0.5)
        self.assertAlmostEqual(result["payment_ratio"], 15_000 / 70_000)
        self.assertEqual(result["n_late_months"], 2)
        self.assertEqual(result["max_delay"], 2)
        self.assertAlmostEqual(result["avg_delay"], 3 / 6)

    def test_client_with_no_bills_counts_as_fully_paid(self):
        result = add_features(pd.DataFrame([make_client()])).iloc[0]

        self.assertEqual(result["payment_ratio"], 1.0)
        self.assertEqual(result["avg_utilization"], 0.0)

    def test_payment_ratio_is_capped_at_one(self):
        client = make_client(bill_1=1_000, pay_amt_1=5_000)

        result = add_features(pd.DataFrame([client])).iloc[0]

        self.assertEqual(result["payment_ratio"], 1.0)

    def test_negative_bill_counts_as_nothing_owed(self):
        client = make_client(bill_1=-5_000)

        result = add_features(pd.DataFrame([client])).iloc[0]

        self.assertEqual(result["avg_utilization"], 0.0)

    def test_invalid_credit_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "limit_bal"):
            add_features(pd.DataFrame([make_client(limit_bal=0)]))

    def test_does_not_need_the_target_and_does_not_modify_input(self):
        df = pd.DataFrame([make_client(bill_1=10_000)]).drop(columns=[TARGET])
        before = df.copy()

        result = add_features(df)

        pd.testing.assert_frame_equal(df, before)
        for feature in ENGINEERED_FEATURES:
            self.assertIn(feature, result.columns)


class TestFeatureColumns(unittest.TestCase):
    def test_excludes_target_and_sex(self):
        df = add_features(pd.DataFrame([make_client()]))

        features = get_feature_columns(df)

        self.assertNotIn(TARGET, features)
        self.assertNotIn("sex", features)
        for feature in ENGINEERED_FEATURES + ["limit_bal", "age", "pay_1"]:
            self.assertIn(feature, features)


class TestSplitData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n = 5_000
        cls.df = pd.DataFrame(
            {"value": rng.normal(size=n), TARGET: (rng.random(n) < 0.22).astype(int)}
        )

    def test_sizes_are_60_20_20(self):
        train, val, test = split_data(self.df)

        self.assertEqual(len(train) + len(val) + len(test), len(self.df))
        self.assertEqual(len(test), 1_000)
        self.assertEqual(len(val), 1_000)
        self.assertEqual(len(train), 3_000)

    def test_parts_do_not_overlap(self):
        train, val, test = split_data(self.df)

        self.assertFalse(set(train.index) & set(val.index))
        self.assertFalse(set(train.index) & set(test.index))
        self.assertFalse(set(val.index) & set(test.index))

    def test_default_rate_is_the_same_in_every_part(self):
        overall = self.df[TARGET].mean()

        for part in split_data(self.df):
            self.assertAlmostEqual(part[TARGET].mean(), overall, delta=0.005)

    def test_same_seed_gives_same_split(self):
        first = split_data(self.df, random_state=1)
        second = split_data(self.df, random_state=1)

        for a, b in zip(first, second):
            self.assertEqual(list(a.index), list(b.index))

    def test_different_seed_gives_different_split(self):
        first, _, _ = split_data(self.df, random_state=1)
        second, _, _ = split_data(self.df, random_state=2)

        self.assertNotEqual(list(first.index), list(second.index))

    def test_invalid_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            split_data(self.df, test_size=0.6, val_size=0.5)

    def test_split_xy(self):
        X, y = split_xy(self.df, ["value"])

        self.assertEqual(list(X.columns), ["value"])
        self.assertEqual(len(X), len(y))


if __name__ == "__main__":
    unittest.main()
