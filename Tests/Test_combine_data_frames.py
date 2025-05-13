import unittest

import numpy as np
import pandas as pd

from Utils.Transformers import combine_dataframes


class TestCombineDataFrames(unittest.TestCase):
    def setUp(self):
        self.df1 = pd.DataFrame({"A": [1, 2, 3, -9999], "B": [-9999, 5, -9999, 7]})
        self.df2 = pd.DataFrame({"A": [-9999, 2, 3, 4], "B": [1, -9999, 3, -9999]})
        self.expected_result = pd.DataFrame(
            {"A": [-9999, 2, 3, -9999], "B": [-9999, -9999, -9999, -9999]}
        )

    def test_combine_dataframes(self):
        result = combine_dataframes(self.df1, self.df2, -9999)
        pd.testing.assert_frame_equal(result, self.expected_result)
