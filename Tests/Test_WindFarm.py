import os
import unittest
from enum import Enum

import numpy as np
import pandas as pd

from Model.WindFarm import WindFarm
from Utils.Enums import AggTypes, DataSourceType

if os.path.basename(os.getcwd()) != "Tests":
    os.chdir("Tests")


class TestWindFarm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # the root path is the project root change so our relative paths are correct

        # Create a sample CSV file with data for three turbines
        cls.csv_data = pd.DataFrame(
            {
                "WAK-T001-GEN-BRG-DE-T-C": [10, 20, 30, 40, 50],
                "WAK-T001-GEN-BRG-NDE-T-C": [15, 25, 35, 45, 55],
                "WAK-T001-GEN-COOLING-T-C": [12, 22, 32, 42, 52],
                "WAK-T001-HS-BRG-T-C": [18, 28, 38, 48, 58],
                "WAK-T001-GEN-SPD-RPM": [1000, 1300, 1200, 1100, 1150],
                "WAK-T001-KW": [1000, 1300, 1200, 1100, 1150],
                "WAK-T001-BLADE-ANGLE-A": [2, 3, 1, 4, 0],
                "WAK-T002-GEN-BRG-DE-T-C": [5, 10, 15, 20, 25],
                "WAK-T002-GEN-BRG-NDE-T-C": [8, 16, 24, 32, 40],
                "WAK-T002-GEN-COOLING-T-C": [7, 14, 21, 28, 35],
                "WAK-T002-GEN-SPD-RPM": [1000, 1300, 1200, 1100, 1150],
                "WAK-T002-KW": [1000, 1300, 1200, 1100, 1150],
                "WAK-T002-BLADE-ANGLE-A": [2, 3, 1, 4, 0],
                "WAK-T002-HS-BRG-T-C": [12, 24, 36, 48, 60],
                "WAK-T003-GEN-BRG-DE-T-C": [2, 4, 6, 8, 10],
                "WAK-T003-GEN-BRG-NDE-T-C": [3, 6, 9, 12, 15],
                "WAK-T003-GEN-COOLING-T-C": [2, 4, 6, 8, 10],
                "WAK-T003-HS-BRG-T-C": [6, 12, 18, 24, 30],
                "WAK-T003-GEN-SPD-RPM": [1000, 1300, 1200, 1100, 1150],
                "WAK-T003-KW": [1000, 1300, 1200, 1100, 1150],
                "WAK-T003-BLADE-ANGLE-A": [2, 3, 1, 4, 0],
            },
            index=pd.date_range("2022-01-01", periods=5, freq="H"),
        )
        cls.csv_data.index.name = "Timestamp"

        cmp_dict = {
            "DateTime": [
                "5/1/2023 12:09:56 AM",
                "5/1/2023 12:19:56 AM",
                "5/1/2023 12:29:56 AM",
                "5/1/2023 12:39:56 AM",
                "5/1/2023 12:49:56 AM",
                "5/1/2023 12:59:56 AM",
            ],
            "WAK-T001-ERR-CODE": [2, 2, 2, 2, 2, 2],
            "DateTime.1": [
                "5/1/2023 12:09:56 AM",
                "5/1/2023 12:19:56 AM",
                "5/1/2023 12:29:56 AM",
                "5/1/2023 12:39:56 AM",
                "5/1/2023 12:49:56 AM",
                "5/1/2023 12:59:56 AM",
            ],
            "WAK-T001-STATE": [16, 16, 16, 16, 16, 16],
            "DateTime.2": [
                "5/1/2023 12:09:56 AM",
                "5/1/2023 12:19:56 AM",
                "5/1/2023 12:29:56 AM",
                "5/1/2023 12:39:56 AM",
                "5/1/2023 12:49:56 AM",
                "5/1/2023 12:59:56 AM",
            ],
            "WAK-T002-ERR-CODE": [2, 2, 2, 2, 2, 2],
            "DateTime.3": [
                "5/1/2023 12:09:56 AM",
                "5/1/2023 12:19:56 AM",
                "5/1/2023 12:29:56 AM",
                "5/1/2023 12:39:56 AM",
                "5/1/2023 12:49:56 AM",
                "5/1/2023 12:59:56 AM",
            ],
            "WAK-T002-STATE": [16, 16, 16, 16, 16, 16],
            "DateTime.4": [
                "5/1/2023 12:09:56 AM",
                "5/1/2023 12:19:56 AM",
                "5/1/2023 12:29:56 AM",
                "5/1/2023 12:39:56 AM",
                "5/1/2023 12:49:56 AM",
                "5/1/2023 12:59:56 AM",
            ],
            "WAK-T003-ERR-CODE": [2, 2, 2, 2, 2, 2],
            "DateTime.5": [
                "5/1/2023 12:09:58 AM",
                "5/1/2023 12:19:58 AM",
                "5/1/2023 12:29:58 AM",
                "5/1/2023 12:39:58 AM",
                "5/1/2023 12:49:58 AM",
                "5/1/2023 12:59:58 AM",
            ],
            "WAK-T003-STATE": [16, 16, 16, 16, 16, 16],
        }

        cmp_data = pd.DataFrame(cmp_dict)
        cmp_data.replace("", np.nan, inplace=True)
        cmp_data["DateTime"] = pd.to_datetime(
            cmp_data["DateTime"], format="%m/%d/%Y %I:%M:%S %p"
        )
        cmp_data["DateTime.2"] = pd.to_datetime(
            cmp_data["DateTime.2"], format="%m/%d/%Y %I:%M:%S %p"
        )
        cmp_data["DateTime.3"] = pd.to_datetime(
            cmp_data["DateTime.3"], format="%m/%d/%Y %I:%M:%S %p"
        )
        cmp_data["DateTime.4"] = pd.to_datetime(
            cmp_data["DateTime.4"], format="%m/%d/%Y %I:%M:%S %p"
        )
        cmp_data["DateTime.5"] = pd.to_datetime(
            cmp_data["DateTime.5"], format="%m/%d/%Y %I:%M:%S %p"
        )
        cls.cmp_data = cmp_data

    @classmethod
    def tearDownClass(cls):
        # Delete the sample CSV file
        # os.remove(cls.csv_file_path)
        pass

    def setUp(self):
        # Create a WindFarm instance with the sample CSV file
        self.wind_farm = WindFarm(
            project="WAK",
            avg_data=self.csv_data,
            compressed_data=self.cmp_data,
            data_source_type=DataSourceType.CSV,
            data_freq="H",
        )

    def test_get_turbine_names(self):
        expected_turbine_names = ["WAK-T001", "WAK-T002", "WAK-T003"]
        actual_turbine_names = self.wind_farm.get_turbine_names()
        self.assertListEqual(actual_turbine_names, expected_turbine_names)

    def test_get_turbine(self):
        expected_turbine_data = {
            "WAK-T002-BLADE-ANGLE-A": [2, 3, 1, 4, 0],
            "WAK-T002-GEN-BRG-DE-T-C": [5, 10, 15, 20, 25],
            "WAK-T002-GEN-BRG-NDE-T-C": [8, 16, 24, 32, 40],
            "WAK-T002-GEN-COOLING-T-C": [7, 14, 21, 28, 35],
            "WAK-T002-GEN-SPD-RPM": [1000, 1300, 1200, 1100, 1150],
            "WAK-T002-HS-BRG-T-C": [12, 24, 36, 48, 60],
            "WAK-T002-KW": [1000, 1300, 1200, 1100, 1150],
        }
        expected_df = pd.DataFrame(
            expected_turbine_data,
            index=pd.date_range("2022-01-01", periods=5, freq="H", name="Timestamp"),
        )

        actual_turbine_data = self.wind_farm.get_turbine("WAK-T002").data
        actual_turbine_data = actual_turbine_data.asfreq("H")

        pd.testing.assert_frame_equal(actual_turbine_data, expected_df)

        expected_turbine_data = {
            "WAK-T003-BLADE-ANGLE-A": [2, 3, 1, 4, 0],
            "WAK-T003-GEN-BRG-DE-T-C": [2, 4, 6, 8, 10],
            "WAK-T003-GEN-BRG-NDE-T-C": [3, 6, 9, 12, 15],
            "WAK-T003-GEN-COOLING-T-C": [2, 4, 6, 8, 10],
            "WAK-T003-GEN-SPD-RPM": [1000, 1300, 1200, 1100, 1150],
            "WAK-T003-HS-BRG-T-C": [6, 12, 18, 24, 30],
            "WAK-T003-KW": [1000, 1300, 1200, 1100, 1150],
        }
        expected_df = pd.DataFrame(
            expected_turbine_data,
            index=pd.date_range("2022-01-01", periods=5, freq="H", name="Timestamp"),
        )
        actual_turbine_data = self.wind_farm.get_turbine("WAK-T003").data
        actual_turbine_data = actual_turbine_data.asfreq("H")
        pd.testing.assert_frame_equal(actual_turbine_data, expected_df)

    def test_get_subset_by_component_type_and_date_range(self):
        expected_subset = pd.DataFrame(
            {
                "WAK-T001-GEN-COOLING-T-C": [12, 22, 32, 42, 52],
                "WAK-T002-GEN-COOLING-T-C": [7, 14, 21, 28, 35],
                "WAK-T003-GEN-COOLING-T-C": [2, 4, 6, 8, 10],
            },
            index=pd.date_range("2022-01-01", periods=5, freq="H"),
        )
        expected_subset.index.name = "Timestamp"

        actual_subset = self.wind_farm.get_subset(
            component_type="Gen_CoolingFluid_Temp",
            start_date="2022-01-01",
            end_date="2022-01-03",
        )
        pd.testing.assert_frame_equal(actual_subset.asfreq("H"), expected_subset)

    def test_get_subset_by_turbine_name(self):
        expected_subset = pd.DataFrame(
            {
                "WAK-T001-BLADE-ANGLE-A": [2, 3, 1, 4, 0],
                "WAK-T001-GEN-BRG-DE-T-C": [10, 20, 30, 40, 50],
                "WAK-T001-GEN-BRG-NDE-T-C": [15, 25, 35, 45, 55],
                "WAK-T001-GEN-COOLING-T-C": [12, 22, 32, 42, 52],
                "WAK-T001-GEN-SPD-RPM": [1000, 1300, 1200, 1100, 1150],
                "WAK-T001-HS-BRG-T-C": [18, 28, 38, 48, 58],
                "WAK-T001-KW": [1000, 1300, 1200, 1100, 1150],
            },
            index=pd.date_range("2022-01-01", periods=5, freq="H"),
        )
        expected_subset.index.name = "Timestamp"

        actual_subset = self.wind_farm.get_subset(turbine_name="WAK-T001")
        pd.testing.assert_frame_equal(actual_subset.asfreq("H"), expected_subset)
