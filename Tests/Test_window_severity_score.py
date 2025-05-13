import unittest
import pandas as pd
import numpy as np
import os

# Assuming the function is in a module named 'severity_calculator'
from Utils.Transformers import calculate_window_severity_with_recovery_threshold

if os.path.basename(os.getcwd()) != "Tests":
    os.chdir("Tests")


class TestRollingSeverityWithRecoveryThreshold(unittest.TestCase):
    def test_all_recovery_above_threshold(self):
        # Test case where all windows meet the threshold
        data = {"A": [1, 2, 3, 4, 5, 6], "B": [2, 3, 4, 5, 6, 7]}
        df = pd.DataFrame(data)
        result = calculate_window_severity_with_recovery_threshold(df, 3, 0.5).astype(
            "float64"
        )
        expected = {
            "A": [np.nan, np.nan, 6, 9, 12, 15],
            "B": [np.nan, np.nan, 9, 12, 15, 18],
        }
        expected_df = pd.DataFrame(expected).astype("float64")
        pd.testing.assert_frame_equal(result, expected_df)

    def test_some_recovery_below_threshold(self):
        # Test case where some windows don't meet the threshold
        data = {"A": [1, np.nan, 3, 4, 5, 6], "B": [2, 3, np.nan, 5, 6, 7]}
        df = pd.DataFrame(data)
        result = calculate_window_severity_with_recovery_threshold(df, 3, 0.66).astype(
            "float64"
        )
        expected = {
            "A": [np.nan, np.nan, np.nan, np.nan, 12.0, 15.0],
            "B": [np.nan, np.nan, np.nan, np.nan, np.nan, 18],
        }
        expected_df = pd.DataFrame(expected)
        pd.testing.assert_frame_equal(result, expected_df)

    def test_no_recovery_meets_threshold(self):
        # Test case where no windows meet the threshold
        data = {"A": [1, np.nan, np.nan, 4, 5, 6], "B": [2, 3, np.nan, np.nan, 6, 7]}
        df = pd.DataFrame(data)
        result = calculate_window_severity_with_recovery_threshold(df, 4, 1.0).astype(
            "float64"
        )
        expected = {"A": [np.nan] * 6, "B": [np.nan] * 6}
        expected_df = pd.DataFrame(expected)
        pd.testing.assert_frame_equal(result, expected_df)


# Add more test cases as necessary

if __name__ == "__main__":
    unittest.main()
