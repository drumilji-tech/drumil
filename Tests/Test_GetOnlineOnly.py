import os
import unittest

import pandas as pd

from Model.WindFarm import WindFarm

if os.path.basename(os.getcwd()) != "Tests":
    os.chdir("Tests")


class TestWindFarm(unittest.TestCase):
    def setUp(self):
        # Create a sample compressed data DataFrame
        self.compressed_data = pd.DataFrame(
            {
                "datetime": [
                    "01/01/2022 12:00:00 AM",
                    "01/01/2022 12:06:00 AM",
                    "01/01/2022 12:10:00 AM",
                    "01/01/2022 12:12:00 AM",
                    "01/01/2022 12:22:00 AM",
                ],
                "WAK-T001-ERR-CODE": [1, 2, 2, 1, 2],
                "datetime.1": [
                    "01/01/2022 12:00:00 AM",
                    "01/01/2022 12:06:00 AM",
                    "01/01/2022 12:10:00 AM",
                    "01/01/2022 12:12:00 AM",
                    "01/01/2022 12:26:00 AM",
                ],
                "WAK-T002-ERR-CODE": [2, 1, 2, 1, 2],
                "datetime.2": [
                    "01/01/2022 12:00:00 AM",
                    "01/01/2022 12:06:00 AM",
                    "01/01/2022 12:10:00 AM",
                    "01/01/2022 12:12:00 AM",
                    "01/01/2022 12:29:00 AM",
                ],
                "WAK-T001-STATE": [16, 0, 16, 16, 16],
                "datetime.3": [
                    "01/01/2022 12:00:00 AM",
                    "01/01/2022 12:06:00 AM",
                    "01/01/2022 12:10:00 AM",
                    "01/01/2022 12:12:00 AM",
                    "01/01/2022 12:21:00 AM",
                ],
                "WAK-T002-STATE": [16, 16, 0, 16, 16],
            }
        )

        # Create a sample 10-minute average DataFrame for PITCH-ANGLE and GEN-SPEED
        self.avg_data = pd.DataFrame(
            {
                "datetime": [
                    "01/01/2022 12:00:00 AM",
                    "01/01/2022 12:10:00 AM",
                    "01/01/2022 12:20:00 AM",
                ],
                "WAK-T001-BLADE-ANGLE-A": [2.0, 90.0, -10.0],
                "WAK-T002-BLADE-ANGLE-A": [1.5, 2.0, 1.0],
                "WAK-T001-GEN-SPD-RPM": [1250, 1400, 800],
                "WAK-T002-GEN-SPD-RPM": [1300, 1350, 1400],
                "WAK-T001-KW": [1250, 1400, 800],
                "WAK-T002-KW": [1300, 1350, 1400],
            }
        )
        self.avg_data.index = pd.to_datetime(self.avg_data["datetime"])
        self.avg_data = self.avg_data.drop("datetime", axis=1)

    def test_get_online_only(self):
        # Create an instance of the WindFarm class with the average data and compressed data
        wind_farm = WindFarm(
            avg_data=self.avg_data, compressed_data=self.compressed_data, project="WAK"
        )

        # Test the get_online_only method
        output_mask, output_data = wind_farm.get_online_only()
        print("test 40 output_mask", output_mask)
        # Define the expected output mask
        expected_mask = pd.DataFrame(
            {
                "WAK-T001": [False, False, False],
                "WAK-T002": [True, False, False],
            },
            index=pd.DatetimeIndex(
                [
                    "01/01/2022 12:00:00 AM",
                    "01/01/2022 12:10:00 AM",
                    "01/01/2022 12:20:00 AM",
                ]
            ),
        )

        # Assert the equality of the output mask and the expected mask
        self.assertTrue(output_mask.equals(expected_mask))


if __name__ == "__main__":
    unittest.main()
