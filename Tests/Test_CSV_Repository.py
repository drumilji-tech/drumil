import sys
import time

sys.path.append("..")
import os
import unittest
from io import StringIO

import pandas as pd

from Model.DataAccess import CSV_Repository


class TestCSVRepository(unittest.TestCase):
    def setUp(self):
        # Create some sample data for testing
        self.data_10min = pd.DataFrame(
            {"valueA": [1, 2, 3, 4, 5], "valueB": [5, 6, 7, 8, 9]},
            index=pd.date_range(start="2022-01-01", periods=5, freq="10T"),
        )

        self.data_10sec = pd.DataFrame(
            {"valueC": [10, 11, 12, 13, 14], "valueD": [15, 16, 17, 18, 19]},
            index=pd.date_range(start="2022-01-01", periods=5, freq="10S"),
        )

        self.data_10min.to_csv("test_10min.csv")

    def test_initialization(self):
        repo = CSV_Repository(data_file_path="test_10min.csv")
        self.assertIsNotNone(repo.data)
        self.assertEqual(len(repo.data.columns), 2)
        self.assertTrue("valueA" in repo.data.columns)

    def test_add_data(self):
        repo = CSV_Repository(data_file_path="test_10min.csv")
        self.assertTrue("valueA" in repo.data.columns)
        self.assertFalse("valueC" in repo.data.columns)

        repo.add_data(self.data_10sec)
        self.assertTrue("valueA" in repo.data.columns)
        # valueC should not be in the main dataframe because its frequency is different
        self.assertFalse("valueC" in repo.data.columns)
        self.assertEqual(len(repo._dataframes), 1)
        # There should be an additional dataframe stored due to different frequency

        new_data = pd.DataFrame(
            {"valueE": [20, 21, 22, 23, 24]},
            index=pd.date_range(start="2022-01-02", periods=5, freq="10T"),
        )

        repo.add_data(new_data)
        # valueE should be added to the main dataframe
        self.assertTrue("valueE" in repo.data.columns)

    def test_get_column_data(self):
        repo = CSV_Repository(data_file_path="test_10min.csv")
        repo.add_data(self.data_10sec)

        data_valueA = repo.get_column_data("valueA")
        self.assertEqual(data_valueA.iloc[0, 0], 1)

        data_valueA_B = repo.get_column_data(["valueA", "valueB"])
        self.assertEqual(len(data_valueA_B.columns), 2)

        data_valueC = repo.get_column_data("valueC", freq="10S")
        self.assertEqual(data_valueC.iloc[0, 0], 10)

    def tearDown(self):
        time.sleep(1)
        if os.path.exists("test_10min.csv"):
            os.remove("test_10min.csv")


if __name__ == "__main__":
    unittest.main()
