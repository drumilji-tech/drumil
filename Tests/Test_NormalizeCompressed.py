import unittest

import pandas as pd
from dateutil.parser import parse

from Utils.Transformers import normalize_compressed
from Utils.Enums import ComponentTypes


class TestNormalizeCompressed(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with data and corresponding datetime columns
        data = {
            "datetime_1": [
                "01/01/2022 12:00:00 AM",
                "01/01/2022 12:04:00 AM",
                "01/01/2022 12:10:00 AM",
                "01/01/2022 12:12:00 AM",
                "01/01/2022 12:22:00 AM",
            ],
            "WAK-T001-ERR-CODE": [1, 2, 2, 2, 1],
            "datetime_2": [
                "01/01/2022 12:02:00 AM",
                "01/01/2022 12:06:00 AM",
                "01/01/2022 12:12:00 AM",
                "01/01/2022 12:23:00 AM",
                "01/01/2022 12:28:00 AM",
            ],
            "WAK-T002-ERR-CODE": [2, 1, 2, 2, 2],
            # Add more datetime and value columns as needed
        }
        self.df = pd.DataFrame(data)

    def test_normalize_compressed_with_mask(self):
        # Define the expected output mask based on the provided test data
        expected_mask = pd.DataFrame(
            [[True, False], [True, True], [False, True]],
            columns=["WAK-T001-ERR-CODE", "WAK-T002-ERR-CODE"],
            index=pd.DatetimeIndex(
                ["2022-01-01 00:00:00", "2022-01-01 00:10:00", "2022-01-01 00:20:00"]
            ),
        )

        expected_mask = expected_mask.rename_axis("timestamp")

        # Test the normalize_compressed function with the sample DataFrame and codes [2]
        output_mask, ___ = normalize_compressed(
            data=self.df, type=ComponentTypes.FAULT_CODE.value, codes=[2]
        )

        self.assertTrue(output_mask.equals(expected_mask))


if __name__ == "__main__":
    unittest.main()
