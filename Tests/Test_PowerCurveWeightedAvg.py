import sys

sys.path.append("../")
from Charts.PowerCurve import (
    calculate_weighted_average_power_by_turbine,
    calculate_weighted_average_power_collapse_bins,
)


import unittest
import pandas as pd
import os
import numpy as np

if os.path.basename(os.getcwd()) != "Tests":
    os.chdir("Tests")


class TestCalculateWeightedAveragePower(unittest.TestCase):
    def setUp(self):
        # Setup any common test data here
        self.df_power = pd.DataFrame(
            {
                "2": [0, 0, 0],
                "3": [25, 27.36638, 30],
                "3.5": [131.5911, 44.62, 152.8151],
                "4": [191.7623, 117.7489, 196.3974],
                "4.5": [314.0377, 187.3821, 302.1528],
                "5": [401.206, 296.468, 546.1737],
                "5.5": [496.1294, 500, 546.1737261],
                "6": [704.2746, 689, 662.3531],
            },
            index=["10/17/2023", "10/18/2023", "10/19/2023"],
        )

        self.df_counts = pd.DataFrame(
            {
                "2": [0, 0, 0],
                "3": [9, 10, 12],
                "3.5": [10, 12, 13],
                "4": [5, 23, 9],
                "4.5": [10, 10, 10],
                "5": [13, 15, 16],
                "5.5": [12, 14, 24],
                "6": [12, 13, 14],
            },
            index=["10/17/2023", "10/18/2023", "10/19/2023"],
        )

    def test_calculate_weighted_average_power_by_turbine(self):
        # Test the calculate_weighted_average_power_by_turbine function
        result, sum_counts = calculate_weighted_average_power_by_turbine(
            self.df_power, self.df_counts
        )

        # Define the expected output
        expected_output = pd.Series(
            {
                "2": float("nan"),  # NaN used to represent #DIV/0!
                "3": 27.69883,
                "3.5": 109.65563714285713,
                "4": 146.881427027027,
                "4.5": 267.8575333,
                "5": 418.2153909,
                "5.5": 521.2344445,
                "6": 684.1343,
            }
        )
        print(result, expected_output)
        # Assertions to validate the result
        pd.testing.assert_series_equal(result, expected_output)

    # def test_calculate_weighted_average_power_collapse_bins(self):
    #     # Test the calculate_weighted_average_power_collapse_bins function
    #     result, sum_counts = calculate_weighted_average_power_collapse_bins(
    #         self.df_power, self.df_counts
    #     )

    #     # Define the expected output
    #     expected_output = pd.DataFrame(
    #         {
    #             2: [np.NaN, np.NaN, np.NaN],
    #             3: [81.10057581, 36.77744374, 93.86383659],
    #             4: [273.2792619, 138.8498616, 252.058113],
    #             5: [446.7692674, 394.7248111, 546.1737261],
    #             6: [704.2746, 689, 662.3531],
    #         },
    #         index=["10/17/2023", "10/18/2023", "10/19/2023"],
    #     )

    #     pd.testing.assert_frame_equal(result, expected_output)


if __name__ == "__main__":
    unittest.main()
