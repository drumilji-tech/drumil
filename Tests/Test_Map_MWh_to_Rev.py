import unittest

import pandas as pd

from Utils.Transformers import map_mwh_to_revenue


class TestMapMWhToRevenue(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.df = pd.DataFrame(
            {"PDK-T001": [10.0, 20.0], "KAY-T002": [30.0, 40.0]},
            index=pd.to_datetime(["2023-09-01 00:00:00", "2023-09-01 01:00:00"]),
        )

        self.revenue_dict = {
            "PDK": {"2023-09-01": {0: 2, 1: 2.5}},
            "KAY": {"2023-09-01": {0: 1.5, 1: 1.8}},
        }

    def test_map_mwh_to_revenue(self):
        result_df = map_mwh_to_revenue(self.df, self.revenue_dict)

        # Expected DataFrame after mapping
        expected_df = pd.DataFrame(
            {"PDK-T001": [20.0, 50.0], "KAY-T002": [45.0, 72.0]},
            index=pd.to_datetime(["2023-09-01 00:00:00", "2023-09-01 01:00:00"]),
        )

        # Assert that the resulting DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(result_df, expected_df)


# If this script is executed, run the tests
if __name__ == "__main__":
    unittest.main()
