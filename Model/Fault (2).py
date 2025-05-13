import numpy as np
import pandas as pd
import gc

from Model.Constants.Trip import NON_TRIP_CODES
from Utils.Transformers import MWh_csv_to_dict, fill_missing_vals_from_ref_column


class FaultAnalysis:
    """
    A class used to analyze fault data and compute various metrics.

    The class provides methods to load, reshape, and analyze fault data. It can compute
    adjusted downtime durations, daily downtime, daily fault counts, daily lost energy, and daily lost revenue.
    The adjusted downtime is based on the presence of fault codes and conditions on active and expected power.
    remaining metrics are all based in the downtime calculation with the exception of count.

    Attributes:
        _reshaped_data (pd.DataFrame): The reshaped version of the fault data.
        data (pd.DataFrame): The raw fault data datetime | fault code | datetime | fault code .....
        avg_data (pd.DataFrame): 10 min averaged data for active and expected power.

    Methods:
        reshaped_data: Returns the reshaped fault data.
        load_data: Loads fault data from a given CSV path.
        load_avg_data: Loads averaged data from a given CSV path.
        find_10min_start: Finds the start of the 10-minute interval for a given timestamp.
        reshape_data: Reshapes the fault data into format amenable to grouping and aggregating by fault, turbine and day.
        daily_downtime: Computes the total downtime for each day, turbine, and fault code.
        daily_fault_count: Computes the daily fault count for each day, turbine, and fault code.
    """

    def __init__(
        self,
        project,
        cmp_data=None,
        cmp_data_path=None,
        avg_data=None,
        avg_data_path=None,
        revenue_per_mwh_path=None,
        downtime_lost_energy_df=None,
    ) -> None:
        """
        Initializes the FaultAnalysis class with data or paths to load the data.

        Args:
            project (str): 3 letter code indicatin the project like 'BTH' or 'WAK'. Is used
                to look up project specific parameters
            cmp_data (pd.DataFrame, optional): The compressed fault data. Defaults to None.
            cmp_data_path (str, optional): Path to the CSV file containing compressed fault data. Defaults to None.
            avg_data (pd.DataFrame, optional): The 10 min averaged data. Defaults to None.
            avg_data_path (str, optional): Path to the CSV file containing 10 min averaged data. Defaults to None.
            downtime_lost_energy_df (pandas.DataFrame, optional): a data frame of previously processed data. Introduced
                to allow reprocessing of daily metrics without having to reprocess the full downtime lost energy metrics.

        Raises:
            ValueError: If neither cmp_data nor cmp_data_path is provided. Similarly for avg_data and avg_data_path.
        """

        self._project = project
        self._downtime_lost_energy = downtime_lost_energy_df

        if cmp_data is None and cmp_data_path is None:
            self.data = None
        else:
            self.data = (
                cmp_data if cmp_data is not None else self.load_data(cmp_data_path)
            )

        if avg_data is None and avg_data_path is None:
            self.avg_data = None
        else:
            self.avg_data = (
                avg_data if avg_data is not None else self.load_avg_data(avg_data_path)
            )
        if revenue_per_mwh_path is None:
            self._revenue_grid = MWh_csv_to_dict(
                file_path="../assets/data/RevenuePerMwh.csv"
            )

        else:
            self._revenue_grid = MWh_csv_to_dict(file_path=revenue_per_mwh_path)

    @property
    def reshaped_data(self):
        """the downtime_lost_energy_data"""
        if self._downtime_lost_energy is None:
            self._downtime_lost_energy = self.calculate_fault_metrics()
            if self._downtime_lost_energy.shape[0] > 0:
                self._downtime_lost_energy["AdjustedStartDateTime"] = pd.to_datetime(
                    self._downtime_lost_energy["AdjustedStartDateTime"]
                )
        return self._downtime_lost_energy

    def load_data(self, data_path):
        """
        Loads compressed fault data from the specified CSV path.

        Args:
            data_path (str): Path to the CSV file containing compressed fault data.

        Returns:
            pd.DataFrame: DataFrame containing the loaded fault data.
        """
        return pd.read_csv(data_path, low_memory=False)

    def load_avg_data(self, avg_data_path):
        """
        Loads averaged data from the specified CSV path.

        Args:
            avg_data_path (str): Path to the CSV file containing 10 min averaged data.

        Returns:
            pd.DataFrame: DataFrame containing the loaded averaged data with the index set to datetime.
        """
        return pd.read_csv(avg_data_path, parse_dates=[0], index_col=[0])

    def find_10min_start(self, timestamp):
        """
        Determines the start time of the 10-minute interval that envelopes a given timestamp.

        Args:
            timestamp (datetime): Input timestamp.

        Returns:
            datetime: Start time of the 10-minute interval.
        """

        # Truncate the minute to the nearest lower ten-minute mark
        start = timestamp.replace(
            minute=(timestamp.minute // 10) * 10, second=0, microsecond=0
        )

        return start

    def calculate_fault_metrics(self):
        """Returns a curated dataset of each fault event and its corresponding Downtime, Lost Energy, and Lost Revenue.

        A Fault event is characterized by the duration from when a fault code first appears to when it switches to another code.
        This function considers periods when a fault is present and the turbine isn't producing power, despite an available wind resource.
        It also captures the "Tail" period post fault resolution when the turbine hasn't resumed power production.

        Returns:
            pd.DataFrame: A reshaped dataframe with columns such as StartDateTime, FaultCode, Turbine, EndDateTime, OriginalDuration,
            OriginalStartDateTime, OriginalEndDateTime, PreAdjustedDuration, NextStartDateTime, AdjustedDuration, AdjustedStartDateTime,
            AdjustedEndDateTime, LostEnergy, and LostRevenue.

        Attributes:
            StartDateTime (datetime): Start of the valid fault event. A valid event has a fault code and lacks power production when expected.
            FaultCode (int): The observed fault code.
            Turbine (str): Turbine with the fault.
            EndDateTime (datetime): End of the valid fault event.
            OriginalDuration (int): Fault event duration without considering power production overlap.
            OriginalStartDateTime (datetime): Unadjusted fault start time.
            OriginalEndDateTime (datetime): Unadjusted fault end time.
            PreAdjustedDuration (int): Fault duration after adjusting for non-production when production is expected.
            NextStartDateTime (datetime): Start time of the next fault event (used mainly for internal calculations).
            AdjustedDuration (int): PreAdjustedDuration plus any detected "Tail" period.
            AdjustedStartDateTime (datetime): Start time post adjustment.
            AdjustedEndDateTime (datetime): End time post adjustment, possibly including a tail period.
            LostEnergy (float): Energy lost in MWh during the AdjustedDuration.
            LostRevenue (float): Lost revenue calculated from lost energy.

        Note:
            The tail period represents the duration post fault resolution when the turbine hasn't resumed production, but should
            have based on expected power. Lost revenue is calculated in dollars ($) considering the time of occurrence.
        """
        error_code_cols = [col for col in self.data.columns if "ERR-CODE" in col]
        datetime_cols = [
            col
            for i, col in enumerate(self.data.columns)
            if "DateTime" in col
            and "ERR-CODE" not in col
            and "-STATE" not in self.data.columns[i + 1]
        ]

        datetime_formats = ["%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"]
        for datetime_format in datetime_formats:
            try:
                for col in datetime_cols:
                    self.data[col] = pd.to_datetime(
                        self.data[col], format=datetime_format
                    )
            except ValueError:
                continue

        df_array = []
        downtime_lost_energy_revenue = pd.DataFrame()
        for dt_col, err_col in zip(datetime_cols, error_code_cols):
            turbine = "-".join(err_col.split("-")[:2])

            df_fault = pd.DataFrame(
                {
                    "StartDateTime": self.data[dt_col],
                    "FaultCode": self.data[err_col].astype("category"),
                    "Turbine": turbine,
                }
            )

            df_fault = df_fault.dropna()
            if len(df_fault) == 0:
                print(f"{turbine} is empty")
                continue
            df_fault = df_fault.sort_values(by="StartDateTime")
            df_fault["EndDateTime"] = self.data[dt_col].shift(-1)

            codes = NON_TRIP_CODES[self._project]
            if len(codes) == 1:
                df_fault = df_fault[df_fault["FaultCode"] != codes[0]]
            else:
                mask = np.isin(df_fault["FaultCode"].values, codes)
                df_fault = df_fault[~mask]

            # compute the very last value since it will be misssing
            df_fault.at[df_fault.index[-1], "EndDateTime"] = df_fault.at[
                df_fault.index[-1], "StartDateTime"
            ].floor("10T") + pd.Timedelta(minutes=10)

            df_fault["OriginalDuration"] = (
                df_fault["EndDateTime"] - df_fault["StartDateTime"]
            ).dt.total_seconds()

            # add th original fault code datetimes for use later
            df_fault["OriginalStartDateTime"] = df_fault["StartDateTime"]
            df_fault["OriginalEndDateTime"] = df_fault["EndDateTime"]

            turbine = "-".join(err_col.split("-")[:2])
            active_power_col = f"{turbine}-KW"
            expected_power_col = f"{turbine}-EXPCTD-KW-CALC"

            valid_intervals = self.avg_data[
                (self.avg_data[active_power_col] <= 0)
                & (self.avg_data[expected_power_col] > 0)
            ].index.tolist()

            # append the 10 min end time of the last start
            valid_intervals.append(valid_intervals[-1] + pd.Timedelta(minutes=10))

            # Merge contiguous valid intervals
            df_intervals = pd.DataFrame({"timestamp": valid_intervals})
            # Find rows where the difference to the previous row isn't 10 minutes
            df_intervals["gap"] = df_intervals["timestamp"].diff() != pd.Timedelta(
                minutes=10
            )
            # Assign groups to contiguous intervals, allowing the identification of contiguous blocks
            df_intervals["group"] = df_intervals["gap"].cumsum()

            # Find start and end for each group or contiguous block
            merged_intervals = df_intervals.groupby("group")["timestamp"].agg(
                ["first", "last"]
            )

            # Adjust 'last' if it's equal to 'first'
            merged_intervals.loc[
                merged_intervals["last"] == merged_intervals["first"], "last"
            ] = merged_intervals["first"] + pd.Timedelta(minutes=10)

            # Convert the result to a list of tuples start_time, end_time of
            # each contigous block of ten min valid power intervals
            merged_valid_intervals = list(
                zip(merged_intervals["first"], merged_intervals["last"])
            )

            all_rows = []
            iter_turbine = None

            for _, row in df_fault.iterrows():
                turbine = row["Turbine"]

                if pd.isna(turbine):
                    continue

                if turbine != iter_turbine:
                    print("      ", turbine)
                    iter_turbine = turbine

                fault_start = row["StartDateTime"]
                fault_end = row["EndDateTime"]

                overlapping_intervals = [
                    (valid_start, valid_end)
                    for valid_start, valid_end in merged_valid_intervals
                    if valid_start < fault_end and valid_end > fault_start
                ]

                for start, end in overlapping_intervals:
                    row_tuple = (
                        row["FaultCode"],
                        turbine,
                        max(start, fault_start),
                        min(end, fault_end),
                        (fault_end - fault_start).total_seconds(),
                        row["OriginalStartDateTime"],
                        row["OriginalEndDateTime"],
                    )
                    all_rows.append(row_tuple)

            columns = [
                "FaultCode",
                "Turbine",
                "StartDateTime",
                "EndDateTime",
                "PreAdjustedDuration",
                "OriginalStartDateTime",
                "OriginalEndDateTime",
            ]

            # Concatenate all the segmented dataframes to get the final segmented_df
            segmented_df = pd.DataFrame(all_rows, columns=columns)

            # Preallocate an array of zeros with the same length as df.
            overlap_durations = np.zeros(len(segmented_df))
            # Initialize arrays for adjusted start and end times with 'NaT'
            adjusted_starts = np.full(
                segmented_df.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]"
            )
            adjusted_ends = np.full(
                segmented_df.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]"
            )

            positive_overlaps_list = []

            # attribute tail loss to originating fault
            for idx, (valid_interval_start, valid_interval_end) in enumerate(
                merged_valid_intervals
            ):
                overlap_start = np.maximum(
                    segmented_df["StartDateTime"].values,
                    np.datetime64(valid_interval_start),
                )
                overlap_end = np.minimum(
                    segmented_df["EndDateTime"].values,
                    np.datetime64(valid_interval_end),
                )

                # Compute the next start times for each row in df
                segmented_df["NextStartDateTime"] = segmented_df["StartDateTime"].shift(
                    -1
                )

                # Adjust overlap_end where the next start time is after the current
                # valid_interval_end and
                condition = (segmented_df["NextStartDateTime"] > valid_interval_end) & (
                    valid_interval_start < overlap_end
                )
                overlap_end = np.where(
                    condition, np.datetime64(valid_interval_end), overlap_end
                )

                # calculate the downtime duration
                current_overlap_durations = (
                    (overlap_end - overlap_start).astype("timedelta64[s]").astype(int)
                )

                # Ensure we only count positive overlaps (negative values indicate no overlap)
                positive_overlaps = current_overlap_durations > 0
                current_overlap_durations[~positive_overlaps] = 0
                positive_overlaps_list.extend(positive_overlaps)

                # Directly assign values to the adjusted_starts and adjusted_ends arrays
                adjusted_starts[positive_overlaps] = overlap_start[positive_overlaps]
                adjusted_ends[positive_overlaps] = overlap_end[positive_overlaps]

                overlap_durations += current_overlap_durations

            segmented_df["AdjustedDuration"] = overlap_durations
            segmented_df["AdjustedStartDateTime"] = adjusted_starts
            segmented_df["AdjustedEndDateTime"] = adjusted_ends

            # Makes sure all have a date
            segmented_df = fill_missing_vals_from_ref_column(
                segmented_df, "AdjustedStartDateTime", "StartDateTime"
            )
            segmented_df = fill_missing_vals_from_ref_column(
                segmented_df, "AdjustedEndDateTime", "EndDateTime"
            )
            segmented_df = fill_missing_vals_from_ref_column(
                segmented_df, "AdjustedDuration", "PreAdjustedDuration"
            )

            if not segmented_df.empty:
                df_array.append(segmented_df)

            downtime_df = pd.concat(df_array, axis=0, ignore_index=True)

            downtime_lost_energy_revenue = self.calculate_lost_energy(
                downtime_df=downtime_df, power_df=self.avg_data
            )

            # calculate revenue
            days = downtime_lost_energy_revenue["AdjustedStartDateTime"].dt.strftime(
                "%Y-%m-%d"
            )
            hours = downtime_lost_energy_revenue["AdjustedStartDateTime"].dt.hour

            revenue_per_mwh = [
                self._revenue_grid.get(self._project, {}).get(day, {}).get(hour, 0)
                for day, hour in zip(days, hours)
            ]

            downtime_lost_energy_revenue["LostRevenue"] = downtime_lost_energy_revenue[
                "LostEnergy"
            ]
            downtime_lost_energy_revenue["LostRevenue"] *= revenue_per_mwh

            del df_fault, df_intervals, merged_intervals
            gc.collect()

        return downtime_lost_energy_revenue

    def calculate_lost_energy(self, downtime_df, power_df):
        """
        Compute the lost energy for all turbines based on their respective downtime start and end times and power data.

        This function calculates the energy loss during downtimes by referencing the difference
        between actual power and expected power for each turbine. It accounts for downtimes that
        don't align perfectly with the 10-minute intervals in the power data by using linear interpolation.

        Args:
            downtime_df (pd.DataFrame): DataFrame containing downtime information. Each row represents a downtime event.
                Must have columns 'AdjustedStartDateTime', 'AdjustedEndDateTime', and 'Turbine'.
            power_df (pd.DataFrame): 10 min average DataFrame containing power data.
                Should have columns for each turbine's 'KW' and 'EXPCTD-KW-CALC' values, with 'DateTime' indicating the timestamp.

        Returns:
            pd.DataFrame: The input downtime_df augmented with an 'EnergyLoss' column
                representing the calculated energy loss for each downtime event.
        """

        downtime_df["AdjustedStartDateTime"] = pd.to_datetime(
            downtime_df["AdjustedStartDateTime"]
        )
        downtime_df["AdjustedEndDateTime"] = pd.to_datetime(
            downtime_df["AdjustedEndDateTime"]
        )

        result_df_list = []

        for turbine in downtime_df["Turbine"].unique():
            turbine_downtime = downtime_df[downtime_df["Turbine"] == turbine].copy()
            relevant_power = power_df[[f"{turbine}-KW", f"{turbine}-EXPCTD-KW-CALC"]]
            relevant_power = relevant_power.sort_index()
            start_times = turbine_downtime["AdjustedStartDateTime"]
            end_times = turbine_downtime["AdjustedEndDateTime"]

            start_indices = (
                relevant_power.index.searchsorted(start_times, side="right") - 1
            )
            end_indices = relevant_power.index.searchsorted(end_times, side="left")

            energy_losses = []
            for i, (start, end) in enumerate(zip(start_indices, end_indices)):
                start_time = start_times.iloc[i]
                end_time = end_times.iloc[i]

                # Compute energy loss for start and end cap
                start_cap_loss, end_cap_loss = self.compute_end_caps_energy_loss(
                    start_time, end_time, relevant_power, turbine
                )

                # For the full duration between start and end times
                actual_power = (
                    relevant_power[f"{turbine}-KW"].iloc[start + 1 : end - 1].values
                )
                expected_power = (
                    relevant_power[f"{turbine}-EXPCTD-KW-CALC"]
                    .iloc[start + 1 : end - 1]
                    .values
                )
                full_loss = (
                    np.sum(expected_power - actual_power) / 6000
                )  # Convert to MWh
                total_loss = start_cap_loss + full_loss + end_cap_loss

                energy_losses.append(total_loss)

            turbine_downtime["LostEnergy"] = energy_losses
            result_df_list.append(turbine_downtime)

        final_result_df = pd.concat(result_df_list, ignore_index=True)
        return final_result_df

    def compute_end_caps_energy_loss(
        self, start_time, end_time, relevant_power, turbine
    ):
        """Compute energy lost at the start and end caps of power data in some interval.

        This function calculates the energy loss for the start and end caps by linearly
        interpolating between two data points in the `relevant_power` dataframe and
        finding the area under the curve of the cap duration. To find the area, we
        integrate using the trapezoidal rule.

        Examples:
            >>> compute_end_caps_energy_loss(start_time=pd.Timestamp('2023-08-01 00:05:00'),
                                             end_time=pd.Timestamp('2023-08-01 00:15:00'),
                                             relevant_power=df,
                                             turbine='PDK-T001')
            (0.0025, 0.003)

        Args:
            start_time (pd.Timestamp): The exact starting time of the fault code event.
            end_time (pd.Timestamp): The exact ending time of the fault code event.
            relevant_power (pd.DataFrame): A dataframe containing ten minute avg active
                and expected power data with a datetime index.
            turbine (str): The turbine identifier used to select the relevant columns
                in the dataframe.

        Returns:
            (tuple of floats): Each tuple looks like `(start_cap_loss, end_cap_loss)`,
                where start_cap_loss and end_cap_loss are in units MegaWatt-hours (MWh).
        """
        # Determine the start and end of the ten-minute interval for start_time
        start_interval = start_time.replace(
            minute=(start_time.minute // 10) * 10, second=0
        )
        end_interval = start_interval + pd.Timedelta(minutes=10)

        if end_time <= end_interval:
            idx_start = relevant_power.index.searchsorted(start_time)
            idx_end = relevant_power.index.searchsorted(end_time) - 1

            # Check for out-of-bounds indices
            if idx_start >= len(relevant_power):
                idx_start = len(relevant_power) - 1
            if idx_end >= len(relevant_power):
                idx_end = len(relevant_power) - 1
            energy_start = (
                relevant_power[f"{turbine}-EXPCTD-KW-CALC"].iloc[idx_start]
                - relevant_power[f"{turbine}-KW"].iloc[idx_start]
            )
            energy_end = (
                relevant_power[f"{turbine}-EXPCTD-KW-CALC"].iloc[idx_end]
                - relevant_power[f"{turbine}-KW"].iloc[idx_end]
            )

            delta_seconds_start_time = (
                start_time.minute * 60 + start_time.second
            ) % 600
            delta_seconds_end_time = (end_time.minute * 60 + end_time.second) % 600

            # Ensure delta_seconds_end_time is computed correctly
            if end_time == end_interval:
                delta_seconds_end_time = 600

            m = (energy_end - energy_start) / 600
            linear_func_start = lambda t: energy_start + m * t

            # Compute the integral using the trapezoidal rule
            start_value = linear_func_start(delta_seconds_start_time)
            end_value = linear_func_start(delta_seconds_end_time)
            area = (
                0.5
                * (start_value + end_value)
                * (delta_seconds_end_time - delta_seconds_start_time)
            )  # kW-seconds

            lost_energy = area / 3600000  # Convert from kW-seconds to MWh

            return lost_energy, 0

        # For the start cap
        start_interval = start_time.replace(
            minute=(start_time.minute // 10) * 10, second=0
        )

        if start_time == start_interval:
            start_cap_loss = 0
        else:
            idx_start = relevant_power.index.searchsorted(start_time) - 1
            idx_end = idx_start + 1

            energy_start = (
                relevant_power[f"{turbine}-EXPCTD-KW-CALC"].iloc[idx_start]
                - relevant_power[f"{turbine}-KW"].iloc[idx_start]
            )
            energy_end = (
                relevant_power[f"{turbine}-EXPCTD-KW-CALC"].iloc[idx_end]
                - relevant_power[f"{turbine}-KW"].iloc[idx_end]
            )

            delta_seconds_from_start = (
                start_time.minute * 60 + start_time.second
            ) % 600
            m = (energy_end - energy_start) / 600
            linear_func_start = lambda t: energy_start + m * t

            # Compute the integral using the trapezoidal rule
            # The area is the average of the function values at the start and end times multiplied by the time duration
            start_value = linear_func_start(delta_seconds_from_start)
            end_value = linear_func_start(600)
            area = (
                0.5 * (start_value + end_value) * (600 - delta_seconds_from_start)
            )  # kW-seconds

            start_cap_loss = area / 3600000  # Convert from kW-seconds to MWh

        # For the end cap
        start_interval = end_time.replace(minute=(end_time.minute // 10) * 10, second=0)
        end_interval = start_interval + pd.Timedelta(minutes=10)

        if end_time == start_interval:
            end_cap_loss = 0
        else:
            idx_start = relevant_power.index.searchsorted(end_time) - 1
            idx_end = idx_start + 1

            # Check if idx_end is out-of-bounds
            if idx_end >= len(relevant_power):
                idx_end = len(relevant_power) - 1

            energy_start = (
                relevant_power[f"{turbine}-EXPCTD-KW-CALC"].iloc[idx_start]
                - relevant_power[f"{turbine}-KW"].iloc[idx_start]
            )
            energy_end = (
                relevant_power[f"{turbine}-EXPCTD-KW-CALC"].iloc[idx_end]
                - relevant_power[f"{turbine}-KW"].iloc[idx_end]
            )

            delta_seconds_end = (end_time.minute * 60 + end_time.second) % 600
            m = (energy_end - energy_start) / 600
            linear_func_end = lambda t: energy_start + m * t
            # Compute the integral using the trapezoidal rule
            start_value = linear_func_end(0)
            end_value = linear_func_end(delta_seconds_end)
            area = 0.5 * (start_value + end_value) * (delta_seconds_end)
            end_cap_loss = area / 3600000  # Convert from kW-seconds to MWh

        return start_cap_loss, end_cap_loss

    def daily_downtime(self):
        """Computes the total downtime in seconds for each day, turbine, and fault code.

        The method aggregates the adjusted downtime durations for each day, turbine, and fault code.

        Returns:
            pd.DataFrame: DataFrame containing columns 'Date', 'Turbine', 'FaultCode', and 'Downtime',
                        representing the total downtime for each combination.
        """
        this_reshaped_data = self.reshaped_data.copy()

        # Extract just the date from the StartDateTime
        this_reshaped_data["Date"] = this_reshaped_data[
            "AdjustedStartDateTime"
        ].dt.normalize()

        # Group by Date, Turbine, and FaultCode and sum the AdjustedDuration
        daily_downtime = (
            this_reshaped_data.groupby(["Date", "Turbine", "FaultCode"])[
                "AdjustedDuration"
            ]
            .sum()
            .reset_index()
        )

        daily_downtime.rename(columns={"AdjustedDuration": "Downtime"}, inplace=True)

        return daily_downtime

    def daily_lost_energy(self, override_reshaped_data=None):
        """
        Computes the total lost_energy MWh each day, turbine, and fault code.
        Args:
            override_reshaped_data (pandas.DatFrame): dataframe optionally paased in. mainly provided for testing/debugging.
                Has the same shape as self.reshaped_data.copy().

        Returns:
            pd.DataFrame: DataFrame containing columns 'Date', 'Turbine', 'FaultCode', and 'LostEnergy',
                        representing the total downtime for each combination.

        """
        if override_reshaped_data is not None:
            this_reshaped_data = override_reshaped_data.copy()
        else:
            this_reshaped_data = self.reshaped_data.copy()

        # Extract just the date from the StartDateTime
        this_reshaped_data["Date"] = this_reshaped_data[
            "AdjustedStartDateTime"
        ].dt.normalize()

        # Group by Date, Turbine, and FaultCode and sum the AdjustedDuration
        daily_lost_energy = (
            this_reshaped_data.groupby(["Date", "Turbine", "FaultCode"])["LostEnergy"]
            .sum()
            .reset_index()
        )

        return daily_lost_energy

    def daily_lost_revenue(self):
        """
        Computes the total lost revenue each day, turbine, and fault code.

        Returns:
            pd.DataFrame: DataFrame containing columns 'Date', 'Turbine', 'FaultCode', and 'LostRevenue',
                        representing the total downtime for each combination.
        """
        this_reshaped_data = self.reshaped_data.copy()

        # Extract just the date from the StartDateTime
        this_reshaped_data["Date"] = this_reshaped_data[
            "AdjustedStartDateTime"
        ].dt.normalize()

        # Group by Date, Turbine, and FaultCode and sum the AdjustedDuration
        daily_lost_energy = (
            this_reshaped_data.groupby(["Date", "Turbine", "FaultCode"])["LostRevenue"]
            .sum()
            .reset_index()
        )

        return daily_lost_energy

    def daily_fault_count(self, trip=True):
        """Computes the daily fault count for each day, turbine, and fault code.

        Args:
            trip (bool): If True, only considers adjusted durations greater than 0.
                        This logic will be replaced by more robust trip/no trip
                        identification in the future.
                        If False, considers all rows. Defaults to True.
                        A 'trip' is when the turbine is caused to go offline,
                        producing no power despite possible available wind resource
                        to do so. The event is directly triggered by the fault code.


        Returns:
            pd.DataFrame: DataFrame containing columns 'Date', 'Turbine', 'FaultCode', and 'FaultCount',
                        representing the number of faults for each combination.
        """
        this_reshaped_data = self.reshaped_data.copy()
        if trip:
            this_reshaped_data = this_reshaped_data[
                this_reshaped_data["AdjustedDuration"] > 0
            ].dropna()
        else:
            this_reshaped_data = this_reshaped_data.dropna()

        # Extract just the date from the StartDateTime
        this_reshaped_data["Date"] = this_reshaped_data[
            "AdjustedStartDateTime"
        ].dt.normalize()

        # Group by Date, Turbine, and FaultCode and count the occurrences
        daily_fault_counts = (
            this_reshaped_data.groupby(["Date", "Turbine", "FaultCode"])
            .size()
            .reset_index(name="FaultCount")
        )

        return daily_fault_counts
