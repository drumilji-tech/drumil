import sys

sys.path.append("..")
import unittest

import numpy as np
import pandas as pd

from Model.WindFarm import CalculatedYawErrorFarmComponent


class TestYawErrorCalculation(unittest.TestCase):
    def setUp(self):
        # Setting up a dummy dataframe for testing
        self.df = pd.DataFrame(
            {
                "KAY-T001-YAW-DIR": [0, 10, 20, 30, 40],
                "KAY-T001-WIND-DIR": [5, 15, 25, 35, 45],
                "KAY-T002-YAW-DIR": [0, -10, -20, -30, -40],
                "KAY-T002-WIND-DIR": [5, 5, 5, 5, 5],
            }
        )

        self.instance = CalculatedYawErrorFarmComponent(
            name="Yaw_Error", project="KAY", technology=None, data=self.df
        )

        self.non_calc_df = pd.DataFrame(
            {
                "WAK-T001-YAW-DIR": [0, 10, 20, 30, 40],
                "WAK-T001-WIND-DIR": [5, 15, 25, 35, 45],
                "WAK-T002-YAW-DIR": [0, -10, -20, -30, -40],
                "WAK-T002-WIND-DIR": [5, 5, 5, 5, 5],
            }
        )

        self.non_calc_instance = CalculatedYawErrorFarmComponent(
            name="Yaw_Error", project="WAK", technology=None, data=self.df
        )

    def test_yaw_error_calculation(self):
        result = self.instance.calculate_yaw_error(self.df, rolling_window_size=2)
        print("ln 24 result", result)
        # Checking shape of result
        self.assertEqual(result.shape, (self.df.shape[0], self.df.shape[1] // 2))

        # Checking column names
        self.assertTrue("KAY-T001-YAW-ERROR" in result.columns)
        self.assertTrue("KAY-T002-YAW-ERROR" in result.columns)

        # Check some calculated values
        # Note: This is a basic check. You might need more detailed checks depending on your requirements.
        self.assertAlmostEqual(result["KAY-T001-YAW-ERROR"].iloc[1], -5, places=2)
        self.assertAlmostEqual(result["KAY-T002-YAW-ERROR"].iloc[1], -10, places=2)

    def test_yaw_error_non_calculation(self):
        "some projects use wind dir direction directly"
        result = self.non_calc_instance.calculate_yaw_error(
            self.non_calc_df, rolling_window_size=2
        )

        # Checking shape of result
        self.assertEqual(
            result.shape, (self.df.shape[0], self.non_calc_df.shape[1] // 2)
        )

        # Checking column names
        self.assertTrue("WAK-T001-YAW-ERROR" in result.columns)
        self.assertTrue("WAK-T002-YAW-ERROR" in result.columns)

        # Check some calculated values
        # Note: This is a basic check. You might need more detailed checks depending on your requirements.
        self.assertAlmostEqual(result["WAK-T001-YAW-ERROR"].iloc[1], 10, places=2)
        self.assertAlmostEqual(result["WAK-T002-YAW-ERROR"].iloc[1], 5, places=2)

    def test_missing_columns(self):
        # Removing a required column
        df_missing = self.df.drop("KAY-T001-WIND-DIR", axis=1)

        with self.assertRaises(ValueError):
            self.instance.calculate_yaw_error(df_missing, rolling_window_size=2)

    def test_invalid_rolling_window_size(self):
        # Rolling window size greater than dataframe length
        with self.assertRaises(ValueError):
            self.instance.calculate_yaw_error(self.df, rolling_window_size=10)

    def test_data_cleaning(self):
        # Adding invalid values to the dataframe
        df_invalid = self.df.copy()
        df_invalid["KAY-T001-YAW-DIR"].iloc[2] = -1100

        result = self.instance.calculate_yaw_error(df_invalid, rolling_window_size=2)
        print(result["KAY-T001-YAW-ERROR"])
        # Check if the invalid value row has been cleaned
        self.assertAlmostEqual(result["KAY-T001-YAW-ERROR"].iloc[1], -5, places=2)


if __name__ == "__main__":
    unittest.main()
