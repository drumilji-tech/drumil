import csv
import operator
import functools
import pandas as pd
import numpy as np
import os
from Model.WindFarm import WindFarm, FarmComponent

from enum import Enum
from Utils.Enums import DataSourceType
from Model.DataAccess import RepositoryFactory
from Utils.Constants import PROJECT_SUBSETS


class InputFileType(Enum):
    AVERAGE = "average"
    COMPRESSED = "compressed"


class Fleet:
    """
    Represents a fleet of wind farms.

    This class provides methods and properties to manage and analyze
    a fleet of wind farms based on various input data files.

    See documentation section 1.5.2.Fleet for further information

    Attributes:
        windfarms (dict): A dictionary of wind farm objects, keyed by their names.
        flagged_turbines (pd.DataFrame): A dataframe containing turbines that have been flagged based on certain criteria.
        daily_severity_scores (pd.DataFrame): A dataframe containing daily severity scores for each wind farm.

    Example:
        >>> fleet = Fleet(avg_dir="path/to/average", cmp_dir="path/to/compressed", yaw_dir="path/to/yaw")
        >>> print(fleet.windfarms)
        {"WindFarmName1": <WindFarm object>, ...}
    """

    def __init__(
        self,
        avg_dir=None,
        cmp_dir=None,
        yaw_dir=None,
        oem_powercurves_path=None,
        single_plant=None,
    ):
        """
        Initializes a Fleet object.

        Args:
            avg_dir (str, optional): Directory path to average data files.
            cmp_dir (str, optional): Directory path to compressed data files.
            yaw_dir (str, optional): Directory path to 10 sec yaw data files.
            oem_power_curve_path(str, optional): path to the warranted power curves all plants
            single_plant (str, optional): 3 letter code. if specified will only run for
                the specified plant
        """
        self._windfarms = {}
        self._compressed_dir = cmp_dir
        self._average_dir = avg_dir
        self._yaw_dir = yaw_dir
        self._average_files = self._build_file_dict(
            self._average_dir, InputFileType.AVERAGE
        )
        self._compressed_files = self._build_file_dict(
            self._compressed_dir, InputFileType.COMPRESSED
        )
        self._yaw_files = self._build_file_dict(
            # input file type is not actually average but this type is
            # handled the same way # as the sampled yaw data. the important
            #  part being that it is a square grid of data
            # with regular timestamps
            self._yaw_dir,
            InputFileType.AVERAGE,
        )
        self._flagged_turbines = None
        self._daily_severity_scores = None
        self._oem_powercurves_path = oem_powercurves_path
        self.create_windfarms(single_plant=single_plant)

    @property
    def windfarms(self):
        """dict: Gets the wind farms in the fleet. Generates
        wind farms if not already created."""

        if not bool(self._windfarms):
            self.create_windfarms()
        return self._windfarms

    @property
    def flagged_turbines(self):
        """pd.DataFrame: Gets the flagged turbines.
        Generates flagged turbines if not already identified."""

        if self._flagged_turbines is None:
            (
                self._flagged_turbines,
                self._daily_severity_scores,
            ) = self.get_flagged_turbines()
        return self._flagged_turbines

    @property
    def daily_severity_scores(self):
        """pd.DataFrame: Gets the daily severity scores.
        Computes scores if not already calculated."""

        if self._daily_severity_scores is None:
            (
                self._flagged_turbines,
                self._daily_severity_scores,
            ) = self.get_flagged_turbines()
        return self._daily_severity_scores

    def _build_file_dict(self, dir, fileType):
        """
        Builds a dictionary of file paths based on the project
        abbreviation discovered in each file.

        Args:
            dir (str): The directory path to the data files.
            fileType (InputFileType): The type of data file.

        Returns:
            dict: A dictionary of file paths, keyed by the project abbreviation.

        Raises:
            ValueError: If more than one project is found per file.

        """
        # define a function to extract the project name from each column name
        # depending on file type
        if fileType == InputFileType.AVERAGE:
            get_project_names = lambda dataframe: list(
                set(
                    [
                        name[:3]
                        for name in dataframe.columns
                        if not any(
                            [name.lstrip()[:3] == y for y in ["Unn", "Dat", "Off"]]
                        )
                    ]
                )
            )
        elif fileType == InputFileType.COMPRESSED:
            get_project_names = lambda dataframe: list(
                set(
                    [
                        name[:3]
                        for i, name in enumerate(dataframe.columns)
                        if (i % 2 == 1)
                        and not any(
                            [name.lstrip()[:3] == y for y in ["Unn", "Dat", "Off"]]
                        )
                    ]
                )
            )

        file_dict = {}
        for file in os.listdir(dir):
            this_frame = pd.read_csv(os.path.join(dir, file), nrows=1)
            projects_found = get_project_names(this_frame)

            if len(projects_found) == 1:
                file_dict[projects_found[0]] = os.path.join(dir, file)
            else:
                raise ValueError(
                    f"Fleet ln 56: {file} was parsed. Only one project should be found per file. {projects_found} found"
                )

        return file_dict

    def create_windfarms(self, single_plant=None):
        """
        For each project file found in the input directories
        create the corresponding windfarm object.

        If a  project has multiple technologies, separate the
        project data in subsets for each technology
        and instantiatiate a windfarm object for
        each subset other wise use all the turbines
        to create the windfarm object.
        """
        for project in self._average_files.keys():
            if single_plant is not None:
                if single_plant != project:
                    continue
            #          if "GSW" in project: # This line is here just to ignore Glass Sands. Comment line to include
            #              continue

            if project not in self._compressed_files:
                print(
                    f"Fleet.create_windfarms : skipping {project}. Average file but No compressed file found"
                )
                continue

            if project in PROJECT_SUBSETS:
                project_avg_data = pd.read_csv(
                    self._average_files[project], index_col=[0], parse_dates=[0]
                )
                project_cmp_data = pd.read_csv(
                    self._compressed_files[project], low_memory=False
                )
                project_yaw_data = pd.read_csv(
                    self._yaw_files[project], index_col=[0], parse_dates=[0]
                )

                for technology, turbines in PROJECT_SUBSETS[project].items():
                    # extract this subset from avg data
                    project_avg_subset_columns = [
                        x
                        for x in project_avg_data.columns
                        if any(y in x for y in turbines)
                    ]
                    project_avg_subset_data = project_avg_data[
                        project_avg_subset_columns
                    ]

                    # extract this subset from yaw data
                    project_yaw_subset_columns = [
                        x
                        for x in project_yaw_data.columns
                        if any(y in x for y in turbines)
                    ]
                    project_yaw_subset_data = project_yaw_data[
                        project_yaw_subset_columns
                    ]

                    cmp_col_to_keep = []
                    # Iterate through the data columns
                    for i in range(1, len(project_cmp_data.columns), 2):
                        col_name = project_cmp_data.columns[i]
                        if any(x in col_name for x in turbines):
                            datetime_col = project_cmp_data.columns[i - 1]
                            cmp_col_to_keep.append(datetime_col)
                            cmp_col_to_keep.append(col_name)

                    project_cmp_subset_data = project_cmp_data.loc[:, cmp_col_to_keep]
                    project_subset_name = f"{project}_{technology}"

                    print(f"Fleet 122: Processing {project_subset_name}...")
                    self._windfarms[project_subset_name] = WindFarm(
                        avg_data=project_avg_subset_data,
                        compressed_data=project_cmp_subset_data,
                        yaw_data=project_yaw_subset_data,
                        data_source_type=DataSourceType.CSV,
                        data_freq="10T",
                        project=project,
                        technology=technology,
                        oem_powercurve_path=self._oem_powercurves_path,
                    )
            else:
                print(f"Fleet 125: Processing {project}...")
                self._windfarms[project] = WindFarm(
                    avg_path=self._average_files[project],
                    compressed_path=self._compressed_files[project],
                    yaw_path=self._yaw_files.get(project),
                    data_source_type=DataSourceType.CSV,  # This could be SQL (if linked to SPC data)
                    data_freq="10T",
                    project=project,
                    oem_powercurve_path=self._oem_powercurves_path,
                )

    def get_daily_efficiency(self):
        """
        Combines daily efficincies from each project and outputs a single dataset.

        Returns:
            (pandas.DataFrame): columns are turbine index is day in the form mm/dd/yyyy
        """

        daily_efficiencies = []
        for wf_name, windfarm in self.windfarms.items():
            this_daily_efficiency = windfarm.Lost_Energy.daily_efficiency
            if this_daily_efficiency is not None:
                daily_efficiencies.append(this_daily_efficiency)

        daily_efficiency = pd.concat(daily_efficiencies, axis=1, join="outer")

        return daily_efficiency

    def get_daily_mean(self):
        """Combines daily means from each component on each project and outputs a single dataset.

        Returns:
            (pandas.DataFrame): columns are turbine-component <project>-<turbine>-<component type>.
                Index is day in the form mm/dd/yyyy.
        """

        daily_means = []
        for wf_name, windfarm in self.windfarms.items():
            for component_name, component_obj in windfarm.components.items():
                if isinstance(component_obj, FarmComponent) and not any(
                    x.lower() in component_name.lower() for x in ["yaw", "dir"]
                ):
                    this_daily_mean = component_obj.daily_mean
                    if this_daily_mean is not None:
                        daily_means.append(this_daily_mean)

        fleet_daily_mean = pd.concat(daily_means, axis=1, join="outer")

        return fleet_daily_mean

    def get_daily_lost_energy(self):
        """
        Combines daily contact lost energy from each project and outputs a single dataset.

        Returns:
            (pandas.DataFrame): columns are turbines, index is day in the form mm/dd/yyyy.
        """

        project_daily_lost_energy = []
        for wf_name, windfarm in self.windfarms.items():
            this_daily_lost_energy = windfarm.Lost_Energy.daily_lost_energy
            if this_daily_lost_energy is not None:
                project_daily_lost_energy.append(this_daily_lost_energy)

        fleet_daily_lost_energy = pd.concat(
            project_daily_lost_energy, axis=1, join="outer"
        )

        return fleet_daily_lost_energy

    def get_daily_lost_revenue(self):
        """
        Combines daily contact lost revenue from each project and outputs a single dataset.

        Returns:
            (pandas.DataFrame): columns are turbines, index is day in the form mm/dd/yyyy.
        """

        project_daily_lost_revenue = []
        for wf_name, windfarm in self.windfarms.items():
            this_daily_lost_revenue = windfarm.Lost_Energy.daily_lost_revenue
            if this_daily_lost_revenue is not None:
                project_daily_lost_revenue.append(this_daily_lost_revenue)

        fleet_daily_lost_revenue = pd.concat(
            project_daily_lost_revenue, axis=1, join="outer"
        )
        fleet_daily_lost_revenue.columns = [
            x.replace("-LOST-ENERGY", "-LOST-REVENUE")
            for x in fleet_daily_lost_revenue.columns
        ]

        return fleet_daily_lost_revenue

    def get_flagged_turbines(
        self, start=None, end=None, period="6H", density_thresh=0.9, n_std=1, top_n=100
    ):
        """Returns a dataframe of flagged turbines for the specified component type, time range, and wind farms.

        Args:
            component_type (str): The component type to filter the flagged turbines. Defaults to None.
            start (str or datetime): The start time of the data range. Defaults to None.
            end (str or datetime): The end time of the data range. Defaults to None.
            windfarms (list): The list of wind farm names to include. Defaults to None.

        Returns:
            pandas.DataFrame: A dataframe of flagged turbines.
        """
        flagged_turbines = []
        severity_scores = []

        for wf_name, windfarm in self.windfarms.items():
            (
                farm_flagged_turbines,
                farm_daily_severity_scores,
            ) = windfarm.get_flagged_turbines(
                start=start,
                end=end,
                period=period,
                density_thresh=density_thresh,
                n_std=n_std,
                top_n=top_n,
            )

            farm_flagged_turbines = farm_flagged_turbines.reset_index()
            farm_flagged_turbines.insert(0, "Wind Farm", wf_name)
            flagged_turbines.append(farm_flagged_turbines)
            severity_scores.append(farm_daily_severity_scores)

        if flagged_turbines:
            fleet_flagged_turbines = pd.concat(flagged_turbines, ignore_index=True)
        else:
            fleet_flagged_turbines = pd.DataFrame()

        if severity_scores:
            fleet_daily_severity_scores = pd.concat(
                severity_scores, axis=1, join="outer"
            )
        else:
            fleet_daily_severity_scores = pd.DataFrame()

        return fleet_flagged_turbines, fleet_daily_severity_scores

    def get_powercurves(self):
        """retreive and combine all power curves from each wind farm

        Returns:
            pandas.DataFrame : dataframe with columns Turbine | Day | 0 m/s | 0.5 m/s | 1.0 m/s | 1.5 m/s |...|
        """

        powercurves = []
        for wf_name, windfarm in self.windfarms.items():
            print(f"Fleet 391: Calculating Power Curves for {wf_name}")
            powercurves.append(windfarm.powercurves)

        fleet_powercurves = pd.concat(powercurves)

        return fleet_powercurves

    def get_powercurve_distributions(self):
        """
            retreive and combine all power curve distributions from each wind farm

        Returns:
            pandas.DataFrame : dataframe with columns Turbine | Day | 0 m/s | 0.5 m/s | 1.0 m/s | 1.5 m/s |...|

        """
        powercurve_distributions = []
        for wf_name, windfarm in self.windfarms.items():
            powercurve_distributions.append(windfarm.powercurve_distributions)

        fleet_powercurve_distributions = pd.concat(powercurve_distributions)

        return fleet_powercurve_distributions
