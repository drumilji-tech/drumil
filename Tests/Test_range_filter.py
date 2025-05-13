import unittest

import numpy as np
import pandas as pd

from Model.Filter import range_filter


class TestRangeFilter(unittest.TestCase):
    def test_range_filter_single_column(self):
        # Arrange
        data = {"A": [1, 2, 3, 4, 5]}
        df = pd.DataFrame(data)
        lower_bound = 2
        upper_bound = 4
        expected_data = {"A": [-9999, 2, 3, 4, -9999]}
        expected_df = pd.DataFrame(expected_data)

        # Act
        filtered_df, stats = range_filter(df, lower_bound, upper_bound)

        # Assert
        pd.testing.assert_frame_equal(filtered_df, expected_df)

    def test_range_filter_multi_column(self):
        # Arrange
        data = {"A": [1, 2, 3, 4, 5], "B": [2, 4, 6, 8, 10]}
        df = pd.DataFrame(data)
        lower_bound = 2
        upper_bound = 7
        expected_data = {"A": [-9999, 2, 3, 4, 5], "B": [2, 4, 6, -9999, -9999]}
        expected_df = pd.DataFrame(expected_data)

        # Act
        filtered_df, stats = range_filter(df, lower_bound, upper_bound)

        # Assert
        pd.testing.assert_frame_equal(filtered_df, expected_df)


if __name__ == "__main__":
    unittest.main()
