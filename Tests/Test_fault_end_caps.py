import unittest
import pandas as pd
import numpy as np
import os

from Model.Fault import FaultAnalysis

if os.path.basename(os.getcwd()) != "Tests":
    os.chdir("Tests")


class TestFaultEndCaps(unittest.TestCase):
    def setUp(self):
        self.fault_analysis = FaultAnalysis(project="PDK")

        self.sample_power_data = {
            "PDK-T001-KW": [1000, 900, 800, 750],
            "PDK-T001-EXPCTD-KW-CALC": [1100, 950, 1100, 1050],
        }
        self.sample_power_df = pd.DataFrame(
            self.sample_power_data,
            index=pd.to_datetime(
                [
                    "2023-01-01 00:00",
                    "2023-01-01 00:10",
                    "2023-01-01 00:20",
                    "2023-01-01 00:30",
                ]
            ),
        )

    def test_irregular_start(self):
        start_time = pd.to_datetime("2023-01-01 00:05")
        end_time = pd.to_datetime("2023-01-01 00:20")
        lost_start, lost_end = self.fault_analysis.compute_end_caps_energy_loss(
            start_time, end_time, self.sample_power_df, "PDK-T001"
        )
        self.assertAlmostEqual(lost_start, 0.005208333333333333, places=3)
        self.assertAlmostEqual(lost_end, 0, places=3)

    def test_irregular_end(self):
        start_time = pd.to_datetime("2023-01-01 00:00")
        end_time = pd.to_datetime("2023-01-01 00:25")
        lost_start, lost_end = self.fault_analysis.compute_end_caps_energy_loss(
            start_time, end_time, self.sample_power_df, "PDK-T001"
        )
        self.assertAlmostEqual(lost_start, 0, places=3)
        self.assertAlmostEqual(lost_end, 0.025, places=3)

    def test_irregular_start_end(self):
        start_time = pd.to_datetime("2023-01-01 00:05")
        end_time = pd.to_datetime("2023-01-01 00:25")
        lost_start, lost_end = self.fault_analysis.compute_end_caps_energy_loss(
            start_time, end_time, self.sample_power_df, "PDK-T001"
        )
        self.assertAlmostEqual(lost_start, 0.005208333, places=3)
        self.assertAlmostEqual(lost_end, 0.025, places=3)

    def test_same_ten_min(self):
        start_time = pd.to_datetime("2023-01-01 00:03")
        end_time = pd.to_datetime("2023-01-01 00:07")
        lost_start, lost_end = self.fault_analysis.compute_end_caps_energy_loss(
            start_time, end_time, self.sample_power_df, "PDK-T001"
        )
        self.assertAlmostEqual(lost_start, 0.005, places=3)
        self.assertAlmostEqual(lost_end, 0, places=3)


# if __name__ == '__main__':
#     unittest.main()
