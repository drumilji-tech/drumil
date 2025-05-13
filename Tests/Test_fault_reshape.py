import os
import unittest
from datetime import datetime

import pandas as pd

from Model.Fault import FaultAnalysis

if os.path.basename(os.getcwd()) != "Tests":
    os.chdir("Tests")


class TestDowntimeCalculator(unittest.TestCase):
    """
    Tests:
    These three cases below are baked into the test data. Even though there is only one assert below,
    the assert tests equality of the calculated dataframe vs. the expected dataframe. In order fo the test
    to pass all 3 of these must have been addressed successfully by the code.

    1. Correctly quantify downtime for  a fault code whose time starts in an invalid range and ends in the
       middle of a valid range. Valid range here meaning valid for the downtime to be quantified or
       when active power is <= 0 and expected pwer is > 0.
    2. Correctly quantify downtime for  a fault code that starts and ends inside of a valid 10 min range.
    3. Correctly quantify downtime for  a fault code that starts exactly on ten min start interval
       and ends exactly on the end of that same 10 min interval
    """

    def setUp(self):
        # Mock data
        cmp_data = pd.DataFrame(
            {
                "DateTime": [
                    "1/1/2023 12:00:00 AM",
                    "1/1/2023 12:12:00 AM",
                    "1/1/2023 12:20:00 AM",
                    "1/1/2023 12:32:02 AM",
                    "1/1/2023 12:34:02 AM",
                    "1/1/2023 12:36:02 AM",
                    "1/1/2023 12:43:00 AM",
                    "1/1/2023 01:05:00 AM",
                ],
                "PDK-T001-ERR-CODE": [160, 150, 170, 180, 190, 200, 210, 220],
            }
        )
        avg_data = pd.DataFrame(
            {
                "PDK-T001-KW": [5, -1, 5, 0, -3, 0, 0, 50],
                "PDK-T001-EXPCTD-KW-CALC": [500, 600, 700, 700, 800, 1000, 1200, 1200],
            },
            index=pd.to_datetime(
                [
                    "1/1/2023 12:00:00 AM",
                    "1/1/2023 12:10:00 AM",
                    "1/1/2023 12:20:00 AM",
                    "1/1/2023 12:30:00 AM",
                    "1/1/2023 12:40:00 AM",
                    "1/1/2023 12:50:00 AM",
                    "1/1/2023 01:00:00 AM",
                    "1/1/2023 01:10:00 AM",
                ],
                format="%m/%d/%Y %I:%M:%S %p",
            ),
        )

        self.calculator = FaultAnalysis(
            project="PDK", cmp_data=cmp_data, avg_data=avg_data
        )

    def test_reshape_data_basic(self):
        reshaped_data = self.calculator.reshaped_data

        # Expected reshaped data based on mock data
        expected_data = pd.DataFrame(
            {
                "StartDateTime": pd.to_datetime(
                    [
                        "1/1/2023 12:10:00 AM",
                        "1/1/2023 12:12:00 AM",
                        "1/1/2023 12:30:00 AM",
                        "1/1/2023 12:32:02 AM",
                        "1/1/2023 12:34:02 AM",
                        "1/1/2023 12:36:02 AM",
                        "1/1/2023 12:43:00 AM",
                        "1/1/2023 01:05:00 AM",
                    ],
                    format="%m/%d/%Y %I:%M:%S %p",
                ),
                "EndDateTime": pd.to_datetime(
                    [
                        "1/1/2023 12:12:00 AM",
                        "1/1/2023 12:20:00 AM",
                        "1/1/2023 12:32:02 AM",
                        "1/1/2023 12:34:02 AM",
                        "1/1/2023 12:36:02 AM",
                        "1/1/2023 12:43:00 AM",
                        "1/1/2023 01:05:00 AM",
                        "1/1/2023 01:10:00 AM",
                    ],
                    format="%m/%d/%Y %I:%M:%S %p",
                ),
                "FaultCode": [160, 150, 170, 180, 190, 200, 210, 220],
                "Turbine": [
                    "PDK-T001",
                    "PDK-T001",
                    "PDK-T001",
                    "PDK-T001",
                    "PDK-T001",
                    "PDK-T001",
                    "PDK-T001",
                    "PDK-T001",
                ],
                "PreAdjustedDuration": [
                    720.0,
                    480.0,
                    722.0,
                    120.0,
                    120.0,
                    418.0,
                    1320.0,
                    300,
                ],
                "AdjustedDuration": [
                    120.0,
                    480.0,
                    122.0,
                    120.0,
                    120.0,
                    418.0,
                    1320.0,
                    300,
                ],
            }
        )

        pd.testing.assert_frame_equal(
            reshaped_data[
                [
                    "StartDateTime",
                    "EndDateTime",
                    "FaultCode",
                    "Turbine",
                    "PreAdjustedDuration",
                    "AdjustedDuration",
                ]
            ],
            expected_data,
            check_exact=False,
        )


if __name__ == "__main__":
    unittest.main()
