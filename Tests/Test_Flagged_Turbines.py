import unittest

import numpy as np
import pandas as pd

from Model.WindFarm import FarmComponent


class TestGetFlaggedTurbinesAlt(unittest.TestCase):
    def setUp(self):
        self.time_index = pd.date_range(
            start="2022-01-01", end="2022-01-10", freq="10T"
        )
        self.data = pd.DataFrame(
            np.random.randn(len(self.time_index), 5),
            index=self.time_index,
            columns=["asset1", "asset2", "asset3", "asset4", "asset5"],
        )

        self.data.name = "DateTime"
        self.fc = FarmComponent(
            project="WAK", name="Gen_Brg_NDE_Temp", data=self.data, freq="10T"
        )

    def test_top_turbines_for_single_asset(self):
        top_turbines, ___ = self.fc.get_flagged_turbines(
            n_std=0, top_n=3, daily_threshold=0.001
        )

        self.assertEqual(len(top_turbines), 3)


if __name__ == "__main__":
    unittest.main()
