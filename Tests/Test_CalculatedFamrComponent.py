import unittest
import os
import pandas as pd
import numpy as np

from Model import WindFarm as wf

if os.path.basename(os.getcwd()) != "Tests":
    os.chdir("Tests")


class TestCalculatedFarmComponent(unittest.TestCase):
    def setUp(self):
        self.interval = "10T"
        self.capacity = 2000

        # Set up the data for 150 intervals (25 hours)
        time_index = pd.date_range(
            start="2022-01-01", periods=150, freq=self.interval, name="DateTime"
        )
        active_power_t001 = np.linspace(1000, 500, 150)
        active_power_t002 = np.linspace(1500, 1000, 150)
        expected_power_t001 = active_power_t001 + 100
        expected_power_t002 = active_power_t002 + 100

        # Replace some values with -9999 to simulate missing data
        active_power_t001[2::20] = -9999
        expected_power_t001[4::20] = -9999
        active_power_t002[2::20] = -9999
        expected_power_t002[3::20] = -9999

        self.data = pd.DataFrame(
            {
                "WAK-T001-KW": active_power_t001,
                "WAK-T001-EXPCTD-KW-CALC": expected_power_t001,
                "WAK-T002-KW": active_power_t002,
                "WAK-T002-EXPCTD-KW-CALC": expected_power_t002,
            },
            index=time_index,
        )
        self.data.index.name = "DateTime"

        self.component = wf.CalculatedFarmComponent(
            project="WAK", name="test", data=self.data, freq=self.interval
        )

    def test_calculate_simple_efficiency(self):
        efficiency = [
            0.883559196,
            -9999,
            -9999,
            -9999,
            -9999,
            -9999,
            -9999,
        ]

        # Run the method under test
        __, output = self.component.calculate_simple_efficiency(
            self.data, self.interval, self.capacity, daily_threshold=0.4
        )

        efficiency_series = pd.Series(efficiency, name="WAK-T001-EFFICIENCY")

        # Now you can use assert_series_equal to compare
        pd.testing.assert_series_equal(
            output["WAK-T001-EFFICIENCY"][-7:].reset_index(drop=True),
            efficiency_series.reset_index(drop=True),
            atol=1e-2,  # Allow a difference of 0.01 in the efficiency values
        )


if __name__ == "__main__":
    unittest.main()
