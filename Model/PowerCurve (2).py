import os
import pandas as pd
import numpy as np

from Utils.Transformers import get_turbine, get_component_type, component_type_map

from Utils.Enums import ComponentTypes


class PowerCurve:
    """Handles calculating and serving power curve information.

    This class corresponds both to the OEM Power Curve itself and the
    measured power curves calculated from each turbine and the farm.
    It provides a base for creating power curves ready to be plotted
    and analyzed.

    Attributes:
        _daily_power_curves (pandas.DataFrame): Daily power curve data. A dataframe
            with columns Turbine|Day|ws_bin_{n}|...where n = 0 to 20.5 m/s
        _daily_distributions (pandas.DataFrame): Daily distributions data.A dataframe
            with columns Turbine|Day|ws_bin_{n}|...where n = 0 to 20.5 m/s
        _data (pandas.DataFrame): Gradient, range, and online filtered air density adjusted
            wind speed and active power data. The file can containothe rnon -used columns so in
            practive the main average file for this project is used.
            The source data for power curve calculation.
        _project_name (str): Three letter project code like 'BTH', 'WAK'
        _oem_power_curve (pandas.DataFrame): OEM power curve data. Optional.

    Args:
        data (pandas.DataFrame): Gradient, range, and online filtered data.
        project_name (str): Three letter project code like "BTH", "WAK", etc.
        oem_power_curve (pandas.DataFrame, optional): OEM power curve data. Defaults to None.

    """

    def __init__(self, data, project_name, oem_power_curve=None):
        self._daily_power_curves = None
        self._daily_distributions = None
        self._data = data
        self._project_name = project_name
        self._oem_power_curve = oem_power_curve
        self._turbine_col_name = "Turbine"
        self._day_col_name = "Day"

    def _merge_final_dataframes(self, frames, index_cols):
        df = pd.concat(frames)
        df.sort_index(level=index_cols, inplace=True)
        return df

    @property
    def daily_power_curves(self):
        """Daily power curve data. A dataframe
        with columns Turbine|Day|ws_bin_n|....."""
        if self._daily_power_curves is None:
            (
                self._daily_power_curves,
                self._daily_distributions,
            ) = self.get_daily_power_curves()
        return self._daily_power_curves

    @property
    def daily_distributions(self):
        """Daily distributions data.A dataframe
        with columns Turbine|Day|ws_bin_n|....."""
        if self._daily_distributions is None:
            (
                self._daily_power_curves,
                self._daily_distributions,
            ) = self.get_daily_power_curves()
        return self._daily_distributions

    def process_turbine_data(
        self, data, turbine, wind_speed_col, active_power_col, bin_width
    ):
        """Processes the turbine data to obtain the grouped data and counts.

        Args:
            data (pandas.DataFrame): Data containing the wind speed and active power columns.
            turbine (str): Name of the turbine. The format looks like {project}-{turbine}, eg. 'PDK-T001'.
            wind_speed_col (str): Name of the column in 'data' containing wind speed data.
            active_power_col (str): Name of the column in 'data' containing active power data.
            bin_width (float): Width in m/s for the Wind Speed bins.

        Returns:
            tuple:
                - grouped (pandas.DataFrame): Data grouped by turbine, day, and wind bin with the mean of active power.
                - turbine_data_counts (pandas.DataFrame): Counts of data points for each group based on turbine, day, and wind bin.

        """
        turbine_data = data[[wind_speed_col, active_power_col]].copy()
        turbine_data.columns = ["wind_speed", "active_power"]
        turbine_data = turbine_data[turbine_data > -1000].dropna()
        turbine_data = turbine_data[turbine_data["active_power"] > 10].dropna()
        turbine_data[self._turbine_col_name] = turbine

        bin_edges = np.arange(-0.25, 20.75, bin_width)
        bin_labels = np.arange(0, 20.5, bin_width)
        turbine_data["wind_bin"] = pd.cut(
            turbine_data["wind_speed"], bins=bin_edges, labels=bin_labels, right=False
        )

        # Add Day column to turbine_data
        turbine_data[self._day_col_name] = turbine_data.index.date

        grouped = (
            turbine_data.groupby(
                [self._turbine_col_name, self._day_col_name, "wind_bin"], observed=True
            )["active_power"]
            .mean()
            .reset_index()
        )
        grouped.columns = [
            self._turbine_col_name,
            self._day_col_name,
            "wind_bin",
            "mean",
        ]

        turbine_data_counts = (
            turbine_data.groupby(
                [self._turbine_col_name, self._day_col_name, "wind_bin"], observed=True
            )
            .size()
            .reset_index(name="count")
        )

        return grouped, turbine_data_counts

    def get_daily_power_curves(self, data=None, bin_width=None):
        """Generates a grid of daily power curves for each turbine.

        This function processes the data to produce daily power curves based on
        the wind speed and active power for each turbine. It also provides an
        option to calculate the park average and includes the OEM power curve
        for comparison.

        Args:
            data (pandas.DataFrame, optional): DataFrame containing air density
                adjusted wind speed and active power. If not provided, the function
                checks the private property and raises an error if data is not
                available either. Default is None.
            bin_width (float, optional): Width in m/s for the Wind Speed bins.
                Default is 0.5.

        Returns:
            tuple:
                - final_power_curve_df (pandas.DataFrame): DataFrame with multi-index
                Turbine|Day and columns of Wind Speed Bin, representing the daily
                power curves.
                - final_dist_df (pandas.DataFrame): DataFrame representing the
                distributions of the power curves.

        Raises:
            ValueError: If no data is passed to the function and no data is found
                from initialization.

        """
        if bin_width is None:
            bin_width = 0.5
        if data is None:
            if self._data is None:
                raise ValueError(
                    """Power Curve: No data was passed in to get_daily_power_curves()
                                    and no data is found from __init__."""
                )
            else:
                data = self._data

        keep_cols = [x for x in data.columns if "Unnamed" not in x]
        data = data[keep_cols]
        turbines = set(data.columns.map(get_turbine))

        ws_tag_extension = component_type_map(
            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value, False
        )[0]
        ap_tag_extension = component_type_map(ComponentTypes.ACTIVE_POWER.value, False)[
            0
        ]

        results = []
        turbine_counts = []
        for turbine in turbines:
            try:
                wind_speed_col = [
                    col
                    for col in data.columns
                    if get_turbine(col) == turbine
                    and get_component_type(col) == ws_tag_extension
                ][0]
            except IndexError:
                raise ValueError(
                    f"No matching column found for turbine {turbine} with type {ws_tag_extension}."
                )
            try:
                active_power_col = [
                    col
                    for col in data.columns
                    if get_turbine(col) == turbine
                    and get_component_type(col) == ap_tag_extension
                ][0]
            except IndexError:
                raise ValueError(
                    f"No matching column found for turbine {turbine} with type {ws_tag_extension}."
                )

            daily_data, daily_counts = self.process_turbine_data(
                data, turbine, wind_speed_col, active_power_col, bin_width
            )
            results.append(daily_data)
            turbine_counts.append(daily_counts)

        combined_data = pd.concat(results)
        combined_counts = pd.concat(turbine_counts)

        final_dist_df = combined_counts.pivot_table(
            index=[self._turbine_col_name, self._day_col_name],
            columns="wind_bin",
            values="count",
        )

        if len(final_dist_df) == 0:
            return None, None
        final_dist_df.columns = final_dist_df.columns.astype(str)
        park_counts = final_dist_df.groupby(level=self._day_col_name).count()
        park_counts.columns = park_counts.columns.astype(str)
        park_counts[self._turbine_col_name] = f"{self._project_name}_Park_Distribution"
        park_counts.set_index(self._turbine_col_name, append=True, inplace=True)
        park_counts = park_counts.swaplevel(0, 1)

        final_df = combined_data.pivot_table(
            index=[self._turbine_col_name, self._day_col_name],
            columns="wind_bin",
            values="mean",
            aggfunc="mean",
        )
        final_df.columns = final_df.columns.astype(str)

        park_average = final_df.groupby(level=self._day_col_name).mean()
        park_average.columns = park_average.columns.astype(str)
        park_average[self._turbine_col_name] = f"{self._project_name}_Park_Average"
        park_average.set_index(self._turbine_col_name, append=True, inplace=True)
        park_average = park_average.swaplevel(0, 1)

        if not isinstance(self._oem_power_curve, pd.DataFrame):
            self._oem_power_curve = self._oem_power_curve.to_frame().T

        oem_power_curve = self._oem_power_curve

        arrays = [[f"{self._project_name} OEM Power Curve"], ["All"]]
        index = pd.MultiIndex.from_arrays(
            arrays, names=(self._turbine_col_name, self._day_col_name)
        )

        if oem_power_curve.shape[0] > 1:
            oem_power_curve = oem_power_curve.T
        oem_power_curve.index = index
        oem_power_curve.columns = oem_power_curve.columns.astype(str)

        final_power_curve_df = self._merge_final_dataframes(
            [final_df, park_average, oem_power_curve],
            [self._turbine_col_name, self._day_col_name],
        )

        final_dist_df = self._merge_final_dataframes(
            [final_dist_df, park_counts], [self._turbine_col_name, self._day_col_name]
        )

        final_power_curve_df.columns = final_power_curve_df.columns.astype(float)
        final_power_curve_df = final_power_curve_df.sort_index(axis=1)

        final_dist_df.columns = final_dist_df.columns.astype(float)
        final_dist_df = final_dist_df.sort_index(axis=1)

        return final_power_curve_df, final_dist_df
