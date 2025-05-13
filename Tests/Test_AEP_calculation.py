import unittest
import pandas as pd
import numpy as np

# Assuming calculate_AEP is in the module named 'aep_module'
from Charts.PowerCurve import calculate_AEP


class TestCalculateAEP(unittest.TestCase):
    def setUp(self):
        # Common setup data can be placed here
        self.project = "PDK"
        self.power_curve_df = pd.DataFrame(
            {self.project: [10, 20, 30, 40]}, index=[0.5, 1.0, 1.5, 2.0]
        )
        self.ws_dist_frame = pd.DataFrame(
            {self.project: [1, 2, 3, 4]}, index=[0.5, 1.0, 1.5, 2.0]
        )

    # def test_calculate_AEP(self):
    #     # Test basic AEP calculation
    #     result = calculate_AEP(self.project, self.power_curve_df, self.ws_dist_frame)

    #     expected_result = (
    #         8760 * ((10 * 1) + (20 * 2) + (30 * 3) + (40 * 4)) / 1000
    #     )  # Example calculation
    #     self.assertAlmostEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
