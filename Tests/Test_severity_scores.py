import os
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from Model.WindFarm import FarmComponent, WindFarm
from Utils.Enums import AggTypes, DataSourceType

if os.path.basename(os.getcwd()) != "Tests":
    os.chdir("Tests")


class FarmComponentTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a sample CSV file with data for three turbines
        cls.csv_data = pd.DataFrame(
            {
                "WAK-T001-GEN-BRG-DE-T-C": [10, 20, 30, 40, 50]
                + [60, 70, 80, 90, 100] * 2,
                "WAK-T001-GEN-SPD-RPM": [1000, 1000, 1050, 1040, 1050]
                + [1060, 1070, 1080, 1090, 1100] * 2,
                "WAK-T001-KW": [1500, 1600, 1150, 1040, 1050]
                + [1660, 1670, 1680, 1690, 1400] * 2,
                "WAK-T001-BLADE-ANGLE-A": [-1, -2, 1, 1.5, 0.4]
                + [-1, -2, 1, 1.5, 0.4] * 2,
                "WAK-T001-GEN-BRG-NDE-T-C": [15, 25, 35, 45, 55]
                + [65, 75, 85, 95, 105] * 2,
                "WAK-T001-GEN-COOLING-T-C": [12, 22, 32, 42, 52]
                + [62, 72, 82, 92, 102] * 2,
                "WAK-T001-HS-BRG-T-C": [18, 28, 38, 48, 58] + [68, 78, 88, 98, 108] * 2,
                "WAK-T002-GEN-BRG-DE-T-C": [5, 10, 15, 20, 25]
                + [30, 35, 40, 45, 50] * 2,
                "WAK-T002-GEN-SPD-RPM": [1000, 1000, 1050, 1040, 1050]
                + [1060, 1070, 1080, 1090, 1100] * 2,
                "WAK-T002-KW": [1500, 1600, 1150, 1040, 1050]
                + [1660, 1670, 1680, 1690, 1400] * 2,
                "WAK-T002-BLADE-ANGLE-A": [-1, -2, 1, 1.5, 0.4]
                + [-1, -2, 1, 1.5, 0.4] * 2,
                "WAK-T002-GEN-BRG-NDE-T-C": [8, 16, 24, 32, 40]
                + [48, 56, 64, 72, 80] * 2,
                "WAK-T002-GEN-COOLING-T-C": [7, 14, 21, 28, 35]
                + [42, 49, 56, 63, 70] * 2,
                "WAK-T002-HS-BRG-T-C": [12, 24, 36, 48, 60]
                + [72, 84, 96, 108, 120] * 2,
                "WAK-T003-GEN-BRG-DE-T-C": [2, 4, 6, 8, 10] + [12, 14, 16, 18, 20] * 2,
                "WAK-T003-GEN-SPD-RPM": [1000, 1000, 1050, 1040, 1050]
                + [1060, 1070, 1080, 1090, 1100] * 2,
                "WAK-T003-KW": [1500, 1600, 1150, 1040, 1050]
                + [1660, 1670, 1680, 1690, 1400] * 2,
                "WAK-T003-BLADE-ANGLE-A": [-1, -2, 1, 1.5, 0.4]
                + [-1, -2, 1, 1.5, 0.4] * 2,
                "WAK-T003-GEN-BRG-NDE-T-C": [3, 6, 9, 12, 15]
                + [18, 21, 24, 27, 30] * 2,
                "WAK-T003-GEN-COOLING-T-C": [2, 4, 6, 8, 10] + [12, 14, 16, 18, 20] * 2,
                "WAK-T003-HS-BRG-T-C": [6, 12, 18, 24, 30] + [36, 42, 48, 54, 60] * 2,
            },
            index=pd.date_range("2022-01-01", periods=15, freq="H"),
        )
        cls.csv_data.index.name = "DateTime"

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
        # the root path is the project root change so our relative paths are correct

        # Create a WindFarm instance with the sample CSV file
        self.wind_farm = WindFarm(
            avg_data=self.csv_data,
            compressed_data=self.cmp_data,
            data_source_type=DataSourceType.CSV,
            data_freq="H",
        )

        self.component = self.wind_farm.Gen_Brg_DE_Temp

    def test_get_severity_scores(self):
        # Set up test parameters
        period = "2H"
        density_thresh = 0.5
        n_std = 1

        # Expected result
        expected_scores = pd.DataFrame(
            data=[None, None, None],
            index=[
                "WAK-T001-GEN-BRG-DE-T-C",
                "WAK-T002-GEN-BRG-DE-T-C",
                "WAK-T003-GEN-BRG-DE-T-C",
            ],
            columns=[pd.to_datetime("2022-01-01")],
        )

        # Calculate severity scores
        scores = self.component.get_severity_scores(
            period=period,
            density_thresh=density_thresh,
            n_std=n_std,
            daily_threshold=0.001,
        )
        scores.index = pd.to_datetime(scores.index)
        # Assert the scores are equal to the expected result
        # Ensure both DataFrames have the same dtype for comparison
        scores = scores.astype("float64")
        expected_scores = expected_scores.astype("float64")
        pd.testing.assert_frame_equal(scores, expected_scores.T)


if __name__ == "__main__":
    unittest.main()
