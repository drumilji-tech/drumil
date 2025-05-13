import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from Model.Filter import gradient_filter


class TestGradientFilter(unittest.TestCase):
    def setUp(self):
        # Create a test dataframe
        self.df = pd.DataFrame(
            {
                "A": [1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.4],
                "B": [2.0, 2.0, 2.0, 2.2, 2.1, 2.0, 1.8, 1.9, 2.0, 2.1],
            }
        )

    def test_order_1_threshold_0_1_consecutive_2(self):
        # Create test dataframe
        df = pd.DataFrame(
            {
                "A": [
                    1.1,
                    1.2,
                    1.4,
                    1.4,
                    1.4,
                    1.4,
                    1.7,
                    1.8,
                    1.9,
                    2.0,
                    2.1,
                    2.2,
                    2.3,
                    2.4,
                    2.5,
                    2.6,
                    2.7,
                    2.8,
                    2.9,
                    3.0,
                ],
                "B": [
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
            },
            pd.date_range("1/1/2021 00:00", periods=20, freq="10T"),
        )

        # Create a copy of the input dataframe
        df_copy = df.copy()

        # Call the function on the copy
        result, stats = gradient_filter(
            df_copy,
            change_threshold=0.09,
            upper_bound=1000,
            lower_bound=-1000,
            repeat_threshold=3,
            gradient_flag_value=-9990,
            diff_depth=1,
            margin=1,
        )

        # Define the expected output
        expected = pd.DataFrame(
            {
                "A": [
                    1.1,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    1.8,
                    1.9,
                    2.0,
                    2.1,
                    2.2,
                    2.3,
                    2.4,
                    2.5,
                    2.6,
                    2.7,
                    2.8,
                    2.9,
                    3.0,
                ],
                "B": [float(-9990)] * 20,
            },
            pd.date_range("1/1/2021 00:00", periods=20, freq="10T"),
        )

        # # Check if the result matches the expected output
        # assert pd.testing.assert_frame_equal(result.astype(float), expected.astype(float))
        # Compare the result to the expected output
        print("test 119 result", result)
        print("test 119 expected", expected)
        np.testing.assert_array_equal(result.values, expected.values)

    def test_order_2_threshold_0_1_consecutive_3(self):
        # Create test dataframe
        df = pd.DataFrame(
            {
                "A": [
                    1.1,
                    1.2,
                    1.3,
                    1.4,
                    1.5,
                    1.6,
                    1.7,
                    1.8,
                    1.9,
                    2.0,
                    2.1,
                    2.2,
                    2.3,
                    2.4,
                    2.5,
                    2.6,
                    2.7,
                    2.8,
                    2.9,
                    3.0,
                ],
                "B": [
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
            },
            pd.date_range("1/1/2021 00:00", periods=20, freq="10T"),
        )

        # Create a copy of the input dataframe
        df_copy = df.copy()

        # Call the function on the copy
        result, stats = gradient_filter(
            df_copy,
            change_threshold=0.09,
            upper_bound=1000,
            lower_bound=-1000,
            repeat_threshold=3,
            gradient_flag_value=-9990,
            diff_depth=2,
            margin=1,
        )

        # Define the expected output
        expected = pd.DataFrame(
            {
                "A": [-9990] * 20,
                "B": [-9990] * 20,
            },
            pd.date_range("1/1/2021 00:00", periods=20, freq="10T"),
        )

        # # Check if the result matches the expected output
        # assert pd.testing.assert_frame_equal(result.astype(float), expected.astype(float))
        # Compare the result to the expected output

        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_10s_interval(self):
        # Create test dataframe with 10-second intervals
        df = pd.DataFrame(
            {
                "A": [
                    1.1,
                    1.2,
                    1.3,
                    5,
                    6,
                    5.5,
                    5.7,
                    1.8,
                    1.9,
                    2.0,
                    2.1,
                    2.2,
                    2.3,
                    2.4,
                    2.5,
                    2.6,
                    2.7,
                    2.8,
                    2.9,
                    3.0,
                ],
                "B": [
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
            },
            pd.date_range("1/1/2021 00:00", periods=20, freq="10S"),
        )

        # Create a copy of the input dataframe
        df_copy = df.copy()

        # Call the function on the copy
        result, stats = gradient_filter(
            df_copy,
            change_threshold=0.09,
            upper_bound=1000,
            lower_bound=-1000,
            repeat_threshold=3,
            gradient_flag_value=-9990,
            diff_depth=2,
            margin=1,
            time_interval="10s",
        )

        # Define the expected output
        expected = pd.DataFrame(
            {
                "A": [
                    1.1,
                    1.2,
                    1.3,
                    5.0,
                    6.0,
                    5.5,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                    -9990.0,
                ],
                "B": [-9990] * 20,
            },
            pd.date_range("1/1/2021 00:00", periods=20, freq="10S"),
        )
        print("result", result)
        print("expected", expected)
        # Compare the result to the expected output
        np.testing.assert_array_almost_equal(result.values, expected.values)
