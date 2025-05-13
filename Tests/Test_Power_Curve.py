import unittest
import pandas as pd
import numpy as np
from Model.PowerCurve import PowerCurve


class TestPowerCurve(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame(
            {
                "BTH-T001-DEN-CPM-WIND-SPD-CALC": [
                    4,
                    4.2,
                    5,
                    5.1,
                    6,
                    6.3,
                    7,
                    7.4,
                    8,
                    8.1,
                ],
                "BTH-T001-KW": [100, 105, 200, 205, 300, 305, 400, 405, 500, 505],
            }
        )
        self.data.index = pd.date_range("10/1/2023", periods=10, freq="10T")
        self.project_name = "BTH"
        self.oem_power_curve = pd.DataFrame(
            {"BTH": [85, 110, 170, 210, 270, 310, 370, 410, 470, 510]},
            index=[4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
        )

        # Create an instance of PowerCurve for testing
        self.power_curve = PowerCurve(
            self.data, self.project_name, self.oem_power_curve
        )

    def test_initialization(self):
        self.assertIsInstance(self.power_curve, PowerCurve)
        self.assertEqual(self.power_curve._data.shape, self.data.shape)
        self.assertEqual(self.power_curve._project_name, self.project_name)
        self.assertEqual(
            self.power_curve._oem_power_curve.shape, self.oem_power_curve.shape
        )

    def test_process_turbine_data(self):
        turbine = "BTH-T001"
        wind_speed_col = "BTH-T001-DEN-CPM-WIND-SPD-CALC"
        active_power_col = "BTH-T001-KW"
        grouped, turbine_data_counts = self.power_curve.process_turbine_data(
            self.data, turbine, wind_speed_col, active_power_col, bin_width=0.5
        )

        self.assertIsInstance(grouped, pd.DataFrame)
        self.assertIsInstance(turbine_data_counts, pd.DataFrame)

        # Check that the wind_speed bins and active_power mean calculations are as expected
        self.assertTrue(
            np.array_equal(grouped["wind_bin"].values, [4, 5, 6, 6.5, 7.0, 7.5, 8.0])
        )
        self.assertTrue(
            np.array_equal(
                grouped["mean"].values, [102.5, 202.5, 300, 305, 400, 405, 502.5]
            )
        )

    def test_get_daily_power_curves(self):
        final_power_curve_df, final_dist_df = self.power_curve.get_daily_power_curves()

        self.assertIsInstance(final_power_curve_df, pd.DataFrame)
        self.assertIsInstance(final_dist_df, pd.DataFrame)

        self.assertTrue(
            np.allclose(
                final_power_curve_df.loc["BTH-T001", :].values[0],
                np.array(
                    [
                        102.5,
                        np.nan,
                        202.5,
                        np.nan,
                        300.0,
                        305.0,
                        400.0,
                        405.0,
                        502.5,
                        np.nan,
                    ]
                ),
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                final_power_curve_df.loc["BTH_Park_Average", :].values[0],
                np.array(
                    [
                        102.5,
                        np.nan,
                        202.5,
                        np.nan,
                        300.0,
                        305.0,
                        400.0,
                        405.0,
                        502.5,
                        np.nan,
                    ]
                ),
                equal_nan=True,
            )
        )

        self.assertTrue(
            np.array_equal(
                final_power_curve_df.loc["BTH OEM Power Curve"].values[0],
                [85, 110, 170, 210, 270, 310, 370, 410, 470, 510],
            )
        )


# Run the tests
if __name__ == "__main__":
    unittest.main()
