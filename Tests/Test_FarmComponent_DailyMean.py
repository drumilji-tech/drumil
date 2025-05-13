import pandas as pd
import unittest
import numpy as np
import sys

sys.path.append("..")
from Model.WindFarm import FarmComponent
from Utils.Enums import ComponentTypes


class TestFarmComponent(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame(
            {
                "PDK-T001": [
                    10,
                    np.nan,
                    30,
                    40,
                    50,
                    np.nan,
                    70,
                    80,
                    90,
                    np.nan,
                    110,
                    120,
                ]
                * 36,
                "PDK-T002": [5, 15, 25, np.nan, 45, 55, 65, 75, 85, 95, 105, 115] * 36,
            },
            index=pd.date_range(start="2023-01-01", periods=12 * 36, freq="10T"),
        )

    def test_daily_mean(self):
        fc = FarmComponent(
            name=ComponentTypes.MAIN_BEARING.value, project="PDK", data=self.sample_data
        )

        expected_daily_mean = self.sample_data.resample("D").mean()
        expected_daily_mean.columns = [x + "_mean" for x in expected_daily_mean.columns]

        pd.testing.assert_frame_equal(fc.daily_mean, expected_daily_mean)

        self.assertEqual(fc.daily_mean.shape, (3, 2))

        np.testing.assert_array_almost_equal(
            fc.daily_mean.iloc[0]["PDK-T001_mean"],
            expected_daily_mean.iloc[0]["PDK-T001_mean"],
        )
        np.testing.assert_array_almost_equal(
            fc.daily_mean.iloc[0]["PDK-T002_mean"],
            expected_daily_mean.iloc[0]["PDK-T002_mean"],
        )


if __name__ == "__main__":
    unittest.main()
