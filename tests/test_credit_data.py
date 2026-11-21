import unittest

import pandas as pd

from credit_data import EXPECTED_COLUMNS, clean_data, rename_columns


def make_frame(*overrides: dict) -> pd.DataFrame:
    """Build a small frame in which every column is 0 unless a row overrides it."""
    rows = []
    for override in overrides:
        row = {column: 0 for column in EXPECTED_COLUMNS}
        row.update({"education": 2, "marriage": 1})
        row.update(override)
        rows.append(row)
    return pd.DataFrame(rows)


class TestCleanData(unittest.TestCase):
    def test_exact_duplicates_are_removed(self):
        df = make_frame({"age": 30}, {"age": 30}, {"age": 40})

        cleaned, report = clean_data(df)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["duplicates_removed"], 1)

    def test_rows_that_differ_only_in_target_are_kept(self):
        df = make_frame({"age": 30, "default": 0}, {"age": 30, "default": 1})

        cleaned, report = clean_data(df)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["duplicates_removed"], 0)

    def test_undocumented_education_codes_become_others(self):
        df = make_frame(
            {"age": 21, "education": 0},
            {"age": 22, "education": 5},
            {"age": 23, "education": 6},
        )

        cleaned, report = clean_data(df)

        self.assertEqual(cleaned["education"].tolist(), [4, 4, 4])
        self.assertEqual(report["education_remapped"], 3)

    def test_undocumented_marriage_code_becomes_others(self):
        df = make_frame({"age": 21, "marriage": 0})

        cleaned, report = clean_data(df)

        self.assertEqual(cleaned["marriage"].tolist(), [3])
        self.assertEqual(report["marriage_remapped"], 1)

    def test_documented_codes_are_unchanged(self):
        df = make_frame(
            {"age": 21, "education": 1, "marriage": 1},
            {"age": 22, "education": 2, "marriage": 2},
            {"age": 23, "education": 3, "marriage": 3},
            {"age": 24, "education": 4, "marriage": 3},
        )

        cleaned, report = clean_data(df)

        self.assertEqual(cleaned["education"].tolist(), [1, 2, 3, 4])
        self.assertEqual(cleaned["marriage"].tolist(), [1, 2, 3, 3])
        self.assertEqual(report["education_remapped"], 0)
        self.assertEqual(report["marriage_remapped"], 0)

    def test_input_frame_is_not_modified(self):
        df = make_frame({"age": 21, "education": 0})

        clean_data(df)

        self.assertEqual(df["education"].tolist(), [0])

    def test_report_row_counts(self):
        df = make_frame({"age": 30}, {"age": 30}, {"age": 40})

        _, report = clean_data(df)

        self.assertEqual(report["rows_before"], 3)
        self.assertEqual(report["rows_after"], 2)


class TestRenameColumns(unittest.TestCase):
    def test_renames_uci_names(self):
        uci_names = ["X1", "X2", "X3", "X4", "X5"] + [
            f"X{i}" for i in range(6, 24)
        ] + ["Y"]
        df = pd.DataFrame([[0] * len(uci_names)], columns=uci_names)

        renamed = rename_columns(df)

        self.assertEqual(list(renamed.columns), EXPECTED_COLUMNS)

    def test_missing_columns_are_reported(self):
        df = pd.DataFrame({"X1": [1], "X2": [2]})

        with self.assertRaisesRegex(ValueError, "Missing columns"):
            rename_columns(df)


if __name__ == "__main__":
    unittest.main()
