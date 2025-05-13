import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from Model.WindFarm import FarmComponent


def create_test_data():
    # Create test data
    data = {
        "A": [-9999, -9999, 3, 4, 5],
        "B": [6, 7, 8, 9, 10],
        "C": [11, 12, 13, 14, 15],
    }
    original_data = pd.DataFrame(data)

    # Create test datamaps
    datamap1 = pd.DataFrame(
        {
            "A": [False, False, True, True, True],
            "B": [True, True, True, True, True],
            "C": [True, True, True, True, True],
        }
    )

    datamap2 = pd.DataFrame(
        {
            "A": [False, True, True, True, True],
            "B": [False, True, True, True, True],
            "C": [False, False, True, True, True],
        }
    )

    return original_data, datamap1, datamap2


class TestDataRemovalStats(unittest.TestCase):
    def test_data_removal_stats(self):
        """
        This tests that the number of newly removed points (False) in
        successive datamaps like (raw, gradient, range, online) are calculated
        and reported correctly
        """
        # Prepare test data
        original_data, datamap1, datamap2 = create_test_data()

        fc = FarmComponent(project="WAK", name="test_component", data=original_data)

        # Expected result
        expected_result = pd.DataFrame(
            {
                "Raw - Percent Missing": [40.0, 0.0, 0.0],
                "Raw - Count Missing": [2, 0, 0],
                "Datamap1 - Percent Removed": [0.0, 0.0, 0.0],
                "Datamap1 - Count Removed": [0, 0, 0],
                "Datamap2 - Percent Removed": [0.0, 20.0, 40.0],
                "Datamap2 - Count Removed": [0, 1, 2],
                "Full Clean - Percent Missing": [40.0, 20.0, 40.0],
                "Full Clean - Count Missing": [2, 1, 2],
            }
        )
        expected_result.index = ["A", "B", "C"]

        actual_result = fc.calculate_data_removal_stats(
            datamap1, datamap2, data_stage_names=["Datamap1", "Datamap2"]
        )

        assert_frame_equal(actual_result, expected_result)


if __name__ == "__main__":
    unittest.main()
