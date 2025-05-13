import unittest
import pandas as pd
import os
from Charts.Treemap import (
    remove_columns_with_missing_values,
)  # Import the function from your module

if os.path.basename(os.getcwd()) != "Tests":
    os.chdir("Tests")


class TestRemoveColumnsWithMissingValues(unittest.TestCase):
    def test_normal_operation(self):
        df = pd.DataFrame(
            {"A": [1, 2, None, 4], "B": [5, 6, 7, 8], "C": [None, None, 11, 12]},
            index=pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ),
        )
        result = remove_columns_with_missing_values(df, "2023-01-01", "2023-01-04")
        self.assertNotIn("A", result.columns)
        self.assertIn("B", result.columns)
        self.assertNotIn("C", result.columns)

    def test_no_start_end_provided(self):
        df = pd.DataFrame(
            {"A": [1, 2, None, 4], "B": [5, None, 7, 8]},
            index=pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ),
        )
        result = remove_columns_with_missing_values(df)
        self.assertNotIn("A", result.columns)
        self.assertNotIn("B", result.columns)

    def test_invalid_start_date(self):
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]},
            index=pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ),
        )
        with self.assertRaises(ValueError):
            remove_columns_with_missing_values(df, "2023-01-05", "2023-01-04")

    def test_invalid_end_date(self):
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]},
            index=pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ),
        )
        with self.assertRaises(ValueError):
            remove_columns_with_missing_values(df, "2023-01-01", "2023-01-05")


# To run the tests
if __name__ == "__main__":
    unittest.main()
