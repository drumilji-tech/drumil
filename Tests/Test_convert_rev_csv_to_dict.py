import unittest
from io import StringIO

import pandas as pd

from Utils.Transformers import MWh_csv_to_dict


class TestMWhCsvToDict(unittest.TestCase):
    def setUp(self):
        # Mock CSV data for testing
        self.csv_data = """Datetime,BR2,BTH
01/01/2021 00:00,10,20
01/01/2021 01:00,15,25"""

        # Create a mock CSV file using StringIO
        self.mock_csv_file = StringIO(self.csv_data)

        # Expected dictionary output
        self.expected_dict = {
            "BR2": {"2021-01-01": {0: 10, 1: 15}},
            "BTH": {"2021-01-01": {0: 20, 1: 25}},
        }

    def test_MWh_csv_to_dict(self):
        # Convert mock CSV to dictionary using the function
        result_dict = MWh_csv_to_dict(self.mock_csv_file)

        # Assert that the resulting dictionary matches the expected dictionary
        self.assertEqual(
            sorted(result_dict.items()), sorted(self.expected_dict.items())
        )


# If this script is executed, run the tests
if __name__ == "__main__":
    unittest.main()
