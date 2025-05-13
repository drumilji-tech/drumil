import unittest
import pandas as pd
import os

if os.path.basename(os.getcwd()) != "Tests":
    os.chdir("Tests")

from Model.Fault import FaultAnalysis


class TestFaultAnalysis(unittest.TestCase):
    def test_calculate_lost_energy(self):
        fault_analysis = FaultAnalysis(project="PDK")

        sample_downtime_data = {
            "Turbine": ["PDK-T001", "PDK-T001"],
            "AdjustedStartDateTime": pd.to_datetime(
                ["2023-01-01 00:05", "2023-01-01 00:20"]
            ),
            "AdjustedEndDateTime": pd.to_datetime(
                ["2023-01-01 00:12", "2023-01-01 00:30"]
            ),
        }
        sample_downtime_df = pd.DataFrame(sample_downtime_data)

        sample_power_data = {
            "PDK-T001-KW": [1000, 900, 800, 750],
            "PDK-T001-EXPCTD-KW-CALC": [1100, 950, 1100, 1050],
        }
        sample_power_df = pd.DataFrame(
            sample_power_data,
            index=pd.to_datetime(
                [
                    "2023-01-01 00:00",
                    "2023-01-01 00:10",
                    "2023-01-01 00:20",
                    "2023-01-01 00:30",
                ]
            ),
        )

        result = fault_analysis.calculate_lost_energy(
            sample_downtime_df, sample_power_df
        )

        expected_losses = [0.007708333333333333, 0.05]

        # Assert that the computed energy losses match the expected losses
        # for i, expected_loss in enumerate(expected_losses):
        self.assertAlmostEqual(
            result.iloc[1]["LostEnergy"], expected_losses[1], places=3
        )


if __name__ == "__main__":
    unittest.main()
