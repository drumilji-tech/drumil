import operator
import sys
import pandas as pd
import numpy as np
import functools
from scipy.stats import zscore
from Model.Turbine import Turbine
from Model.Component import Component
from Model.Constants.General import (
    CLEANING_ORDER,
    TURBINE_TECH,
    COMPRESSED_COLUMN_TYPES,
    DO_NOT_CALCULATE_SEVERITY,
)
from Model.Constants.GradientFilter import GRADIENT_FILTER_PARAMETERS
from Model.Constants.OnlineFilter import ONLINE_FILTER_PARAMETERS
from Model.Constants.RangeFilter import RANGE_FILTER_PARAMETERS
from Model.DataAccess import RepositoryFactory
from Model.PowerCurve import PowerCurve
from Utils.Enums import AggTypes, ComponentTypes, DataSourceType
from Utils.Constants import (
    DEFAULT_PARSE_FUNCS,
    PROJECT_SUBSETS,
)
from Utils.Transformers import (
    get_component_types,
    component_type_map,
    merge_csv_files,
    normalize_compressed,
    MWh_csv_to_dict,
    map_mwh_to_revenue,
    does_precompute_yaw_error,
    get_turbine,
    get_component_type,
    calculate_window_severity_with_recovery_threshold,
    filter_days,
    custom_sum,
)

from Model.Filter import gradient_filter, range_filter


class WindFarm:
    """A class representing a wind farm.

    The WindFarm constructor initializes a new WindFarm object.

    Further information in docs section 1.5.2.WindFarm

    Args:
        path: The path to the input data file. Defaults to None.
        data_source_type: The type of data source to use. Currently supports 'csv' and 'parquet'.
            Defaults to None.
        data_freq: The time frequency of the data, in pandas offset string format. Defaults to '10T'.

    Attributes:
        _turbines: A dictionary of Turbine objects, keyed by the turbine name.
        _data_freq: The time frequency of the data.
        _turbine_name_func: The function used to parse the turbine name from the column names in the input data.
        _component_type_func: The function used to parse the component type from the column names in the input data.
        _components: A dictionary of FarmComponent objects, keyed by the normalized component type name.
        repository: An instance of a data repository, used to retrieve data from the input data file.

    Returns:
        A new WindFarm object.
    """

    def __init__(
        self,
        avg_path=None,
        avg_data=None,
        compressed_path=None,
        compressed_data=None,
        yaw_path=None,
        yaw_data=None,
        data_source_type=DataSourceType.CSV,
        data_freq="10T",
        project="WAK",  # This is just a default value, but it runs for all sites
        technology=None,
        revenue_grid=None,
        oem_powercurve_path=None,
    ):
        """Initializes a new instance of the WindFarm class.
            For each distinct component type parsed from the column names in the input
            file get a normalized component type name and create a property of type FarmComponent
            with that normalized component type name. component_type strings parsed from the columns that do not
            have a mapping in component_type_map will not create a property
            each property is a FarmComponent Object which has available from it the park mean,std, max,min, median
            as available properties. these are fully cleaned. also available on each FarmComponent object are the
            raw data, gradient cleaned data and stats, range cleaned data and stats, online only data and stats.

        Args:
            avg_path: The path to the input data file. Defaults to None.Can be a string or a list. if
                a list then the files are opened individually and merged together using an outer join.
            avg_data: An alternative to passing the path of the average data file. can pass in as a
                dataframe instead.
            compressed_path (str): path to the compressed data file
            compressed_data(pandas.DataFrame): a dataframe of compressed data. if this is passed compressed path is ignored.
            yaw_path: path to yaw data.
            yaw_data: dataframe of yaw data.
            data_source_type: The type of data source to use. Currently supports 'csv' and 'parquet'.
                Defaults to None.
            data_freq: The time frequency of the avg data. compressed data will be processed to this frequency
            and analyzed. offset string format. Defaults to '10T'.
            project (str): three capital letters indicating the project like "WAK". during dev if project
                         is None it is assumed to be WAK.
            technology (str): can be left blank or populated. If populated it matches the project,
                              technology keys found in constants
            revenue_grid():
            oem_powercurve_path (pandas.DataFrame): the path to a dataframe containing all oem power curves for all
                projects and technologies

        """

        self._data_freq = data_freq
        self._compressed_path = compressed_path
        self._compressed_data = compressed_data
        self._yaw_path = yaw_path
        self._yaw_data = yaw_data
        self._oem_powercurve_path = oem_powercurve_path
        self.name = project
        self.technology = technology

        self._turbine_name_func = get_turbine
        self._component_type_func = DEFAULT_PARSE_FUNCS["component_type_func"]
        self._tag_type_suffixes = None
        self._online_parameters = None
        self._online_input_data = None
        self._online_map = None
        self._oem_powercurve = None
        self._powercurves = None
        self._powercurve_distributions = None
        self._components = {}
        self._revenue_dict = (
            revenue_grid
            if revenue_grid is not None
            else MWh_csv_to_dict(file_path="../assets/data/RevenuePerMwh.csv")
        )
        self._turbines = None

        if all(x is None for x in [self._compressed_path, self._compressed_data]):
            raise ValueError(
                "WindFarm.py 90 you must pass either the path to a compressed data file or directly pass in a compressed dataframe. "
            )

        # avg path is a string or a list of strings. either way return a single dataframe
        # assumes each file has the same date range of data
        if avg_data is None:
            if isinstance(avg_path, list):
                input_data = merge_csv_files(avg_path)
            else:
                input_data = pd.read_csv(avg_path, index_col=[0], parse_dates=[0])

                # Convert all columns to numeric, coercing errors to NaN, then fill NaN with -9999
                input_data = input_data.apply(
                    lambda x: pd.to_numeric(x, errors="coerce")
                ).fillna(-9999)

        else:
            input_data = avg_data

        self.repository = RepositoryFactory.create_repository(
            data_source_type=data_source_type, data=input_data
        )

        yaw_input_data = None
        if yaw_data is not None:
            yaw_input_data = yaw_data
        elif yaw_path is not None:
            if isinstance(yaw_path, list):
                yaw_input_data = merge_csv_files(yaw_path)
            else:
                yaw_input_data = pd.read_csv(yaw_path, index_col=[0], parse_dates=[0])

        if yaw_input_data is not None:
            self.repository.add_data(yaw_input_data, freq="10s")

        # Public properties for each measured component type
        # component_type becomes a FarmComponent object
        for component_type, component_obj in self.components.items():
            setattr(
                WindFarm,
                component_type,
                property(
                    lambda self, component_type=component_type: self._get_component(
                        component_type
                    )
                ),
            )

        # check to see if active power and expected power are in the input
        if all(
            component_type in self._components
            for component_type in [
                ComponentTypes.ACTIVE_POWER.value,
                ComponentTypes.EXPECTED_POWER.value,
            ]
        ):
            # add calculated farm components
            # becomes a CalculatedFarmComponent
            self._components[
                ComponentTypes.LOST_ENERGY.value
            ] = CalculatedFarmComponent(
                name=ComponentTypes.LOST_ENERGY.value,
                project=self.name,
                technology=self.technology,
                data=pd.concat(
                    [
                        getattr(self, ComponentTypes.ACTIVE_POWER.value).clean_data,
                        getattr(self, ComponentTypes.EXPECTED_POWER.value).clean_data,
                    ],
                    axis=1,
                ),
                freq="10T",
                revenue_dict=self._revenue_dict,
            )
            # Public properties for each calculated component type
            for component_type, component_obj in self._components.items():
                if not isinstance(component_obj, CalculatedFarmComponent):
                    continue
                setattr(
                    WindFarm,
                    component_type,
                    property(
                        lambda self, component_type=component_type: self._get_component(
                            component_type
                        )
                    ),
                )
        if yaw_input_data is not None:
            # check to see if active power and expected power are in the input
            if does_precompute_yaw_error(self.name):
                required_components = [
                    ComponentTypes.NACELLE_WIND_DIRECTION.value,
                ]
                yaw_data = getattr(
                    self, ComponentTypes.NACELLE_WIND_DIRECTION.value
                ).clean_data
            else:
                required_components = [
                    ComponentTypes.NACELLE_WIND_DIRECTION.value,
                    ComponentTypes.YAW_POSITION.value,
                ]
                yaw_data = pd.concat(
                    [
                        getattr(self, ComponentTypes.YAW_POSITION.value).clean_data,
                        getattr(
                            self, ComponentTypes.NACELLE_WIND_DIRECTION.value
                        ).clean_data,
                    ],
                    axis=1,
                )

            if all(
                component_type in self._components
                for component_type in required_components
            ):
                # add calculated farm components
                self._components[
                    ComponentTypes.YAW_ERROR.value
                ] = CalculatedYawErrorFarmComponent(
                    name=ComponentTypes.YAW_ERROR.value,
                    project=self.name,
                    technology=self.technology,
                    data=yaw_data,
                    freq="10s",
                )
                # Public properties for each calculated yaw error component type
                for component_type, component_obj in self._components.items():
                    if not isinstance(component_obj, CalculatedYawErrorFarmComponent):
                        continue
                    setattr(
                        WindFarm,
                        component_type,
                        property(
                            lambda self, component_type=component_type: self._get_component(
                                component_type
                            )
                        ),
                    )

        # process additional data needed to support filtering and analaysis
        # but not needed to be shown as farm components (on the dashboard)

    def _get_component(self, component_type):
        return self._components[component_type]

    def _create_farm_component(self, component_type):
        """
        create different farm component objects depending on component type
        """
        if any(
            component_type == x
            for x in [
                ComponentTypes.NACELLE_WIND_DIRECTION.value,
                ComponentTypes.YAW_ERROR.value,
                ComponentTypes.YAW_POSITION.value,
            ]
        ):
            return YawFarmComponent(
                name=component_type,
                project=self.name,
                technology=self.technology,
                online_map=self.online_map.copy(),
                data=self.get_subset(component_type=component_type),
            )
        else:
            return FarmComponent(
                name=component_type,
                project=self.name,
                technology=self.technology,
                online_map=self.online_map.copy(),
                data=self.get_subset(component_type=component_type),
            )

    @property
    def online_parameters(self):
        """
        Property that returns the online filtering parameters for the project.

        Returns:
            dict: Online filtering parameters for the project.
        """
        if self.name in PROJECT_SUBSETS:
            return_val = ONLINE_FILTER_PARAMETERS[self.name].get(self.technology)
            if return_val is None:
                print(
                    "WindFarm 294 ",
                    self.name,
                    self.technology,
                    "not found in ",
                    PROJECT_SUBSETS,
                )
            return return_val
        else:
            # we know there are no subsets, get the first item in the dict and take the 2nd element (value) of the
            # key, value tuple that is returned , which is the param dict

            return list(ONLINE_FILTER_PARAMETERS[self.name].items())[0][1]

    @property
    def online_map(self):
        """The combined boolean map produced after filtering by all online parameters"""
        if self._online_map is None:
            self._online_map, self._online_input_data = self.get_online_only()
        return self._online_map

    @property
    def components(self):
        """FarmComponents belonging to this WindFarm"""
        if len(self._components) == 0:
            component_types, tag_suffixes = self.get_component_types()

            self._tag_type_suffixes = tag_suffixes
            for component_type in component_types:
                self._components[component_type] = self._create_farm_component(
                    component_type=component_type
                )
        return self._components

    @property
    def turbines(self):
        """turbine objects belonging to this WindFarm"""
        if self._turbines is None:
            self._turbines = self.get_turbines()
        return self._turbines

    @property
    def oem_powercurve(self):
        """The Warranted Power Curve from the Manufacturer"""
        if self._oem_powercurve is None:
            if self._oem_powercurve_path is None:
                raise ValueError(
                    "WindFarm.oem_powercurve: To run power curves you must pass in the path to a csv file containing all OEM Power curves."
                )
            try:
                self._oem_powercurve = pd.read_csv(
                    self._oem_powercurve_path, index_col=[0]
                )[self.name]
            except KeyError:
                self._oem_powercurve = pd.read_csv(
                    self._oem_powercurve_path, index_col=[0]
                )[f"{self.name}_{self.technology}"]

        return self._oem_powercurve

    @property
    def powercurves(self):
        """retreive or calculate daily power curves for this project

        The power curves use the wind speed measured by the anemometer on the nacelle
        and the active power measured from the turbine. The method of bins is used
        to create curves of average power at each .5 m/s bin center.

        Returns:
            pandas.DataFrame: a dataframe containing power curves for each turbine and each day.
                The Frame columns are laid out in the following way:

                 Turbine | Day | 0 | 0.5 | 1.0 | 1.5 | ... | 40 |

                 each bin is centered, meaning that if I call bin 1.0 that is the average power
                 where wind speed is greater than or equal to .75 m/s and less than 1.25 m/s.
        """
        if self._powercurves is None:
            # compile the clean ws and power data
            try:
                input_data = pd.concat(
                    [
                        self._components[
                            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED.value
                        ].clean_data,
                        self._components[ComponentTypes.ACTIVE_POWER.value].clean_data,
                    ],
                    axis=1,
                )
            except KeyError:
                print(
                    f"WindFarm.powercurve_distribution : No clean data found to construct power curves for {self.name}"
                )
                self._powercurves = pd.DataFrame()
                self._powercurve_distributions = pd.DataFrame()
                return self._powercurves

            power_curve = PowerCurve(
                data=input_data,
                project_name=self.name,
                oem_power_curve=self.oem_powercurve,
            )
            self._powercurves = power_curve.daily_power_curves
            self._powercurve_distributions = power_curve.daily_distributions

        return self._powercurves

    @property
    def powercurve_distributions(self):
        """retreive or calculate daily power curve distributions or bin counts for this project

        The power curves use the wind speed measured by the anemometer on the nacelle
        and the active power measured from the turbine. The method of bins is used
        to create curves of average power at each .5 m/s bin center.  The distributions contain the valid data counts that
        of the data that create the binned average power curves.

        Returns:
            pandas.DataFrame: a dataframe containing power curve counts by bin for each turbine and each day.
                The columns are laid out in the following way:

                 Turbine | Day | 0 | 0.5 | 1.0 | 1.5 | ... | 40 |

                 each bin is centered, meaning that if I call bin 1.0 that is the count of clean data
                 where wind speed is greater than or equal to .75 m/s and less than 1.25 m/s.
        """
        if self._powercurve_distributions is None:
            try:
                input_data = pd.concat(
                    [
                        self._components[
                            ComponentTypes.NACELLE_AD_ADJ_WIND_SPEED
                        ].clean_data,
                        self._components[ComponentTypes.ACTIVE_POWER].clean_data,
                    ],
                    axis=1,
                )
            except KeyError:
                print(
                    f"WindFarm.powercurve_distribution : No clean data found to construct power curve distribution for {self.name}"
                )
                self._powercurves = pd.DataFrame()
                self._powercurve_distributions = pd.DataFrame()
                return self._powercurve_distributions

            power_curve = PowerCurve(
                data=input_data,
                project_name=self.name,
                oem_power_curve=self._oem_powercurve,
            )
            self._powercurves = power_curve.daily_power_curves
            self._powercurve_distributions = power_curve.daily_distributions

        return self._powercurve_distributions

    def get_subset(
        self,
        data=None,
        turbine_name=None,
        column_name=None,
        component_type=None,
        start_date=None,
        end_date=None,
    ):
        """Returns a subset of the data based on the specified column name, component type, and date range.

        Args:
            turbine_name: The name or list of names of the turbine's data to retrieve. Defaults to None.
            column_name: The name or list of names of the column to retrieve. Defaults to None.
            component_type: The type or list of types of component to retrieve. Defaults to None.
            start_date: The start date of the data to retrieve. Defaults to None.
            end_date: The end date of the data to retrieve. Defaults to None.

        Returns:
            pandas.DataFrame: A subset of the data.
        """
        if sum(x is not None for x in [column_name, component_type, turbine_name]) > 1:
            raise ValueError(
                "Only one of column_name, component_type, or turbine_name should be specified."
            )

        base_columns = self.repository.get_all_column_names()

        if turbine_name is not None:
            if not isinstance(turbine_name, list):
                turbine_name = [turbine_name]
            columns = [
                col
                for col in base_columns
                if self._turbine_name_func(col) in turbine_name
            ]

        if component_type is not None:
            if not isinstance(component_type, list):
                component_type = [component_type]

            component_type_search_str = []
            # convert component type property names to searchable tag segments
            for c in component_type:
                mapped_value = component_type_map(key_str=c, rtn_property_str=False)
                if mapped_value is None:
                    component_type_search_str.append(c)
                elif isinstance(mapped_value, list):
                    component_type_search_str.extend(mapped_value)
                else:
                    component_type_search_str.append(mapped_value)

            columns = [
                c
                for c in base_columns
                if any(
                    x == DEFAULT_PARSE_FUNCS["component_type_func"](c)
                    for x in component_type_search_str
                )
            ]

        if column_name is not None:
            if isinstance(column_name, str):
                column_name = [column_name]
            columns = [c for c in base_columns if c in column_name]

        repo_data = self.repository.get_column_data(columns)
        if start_date is None:
            start_date = repo_data.index[0]
        if end_date is None:
            end_date = repo_data.index[-1]

        return_data = repo_data.loc[start_date:end_date]

        return return_data

    def get_online_only(self):
        """
        Calculates a 10 minute boolean dataframe, one column per turbine, that indicates that the turbine is online (True)
        or offline (False). Data recived is a combination of irregular time intervals and 10 minute data. the irregular
        interval data is first converted to regular intervaled data and then normal filtering is applied to yield the final
        booolean dataframe.

        Args:
            None
        Returns:
            pandas.DataFrame of all boolean values, one column per turbine and 10 minute time intervals.

        """
        if self._compressed_data is None:
            compressed_data = pd.read_csv(self._compressed_path, low_memory=False)
        else:
            compressed_data = self._compressed_data

        # iterate data type enums stored for online filtering for this project
        this_project_online_params = self.online_parameters

        # get compressed types for this project
        this_project_compressed_types = COMPRESSED_COLUMN_TYPES[self.name]

        online_single_param_masks = []
        data_list = []

        for type_enum, param_dict in this_project_online_params.items():
            if type_enum in this_project_compressed_types:
                this_mask, this_data = normalize_compressed(
                    data=compressed_data,
                    type=type_enum,
                    codes=param_dict["normal_codes"],
                )

            else:
                this_data = self.get_subset(component_type=type_enum)
                # for now all non compressed columns have an upper_bound
                # and # lower_bound property
                this_mask = (this_data >= param_dict["lower_bound"]) & (
                    this_data < param_dict["upper_bound"]
                )

            online_single_param_masks.append(this_mask)
            data_list.append(this_data)

            # this_mask comes back with full col names
            # trim columns  to turbine trunks only
            # so masks can be combined correctly
            for mask in online_single_param_masks:
                mask.columns = [self._turbine_name_func(col) for col in mask.columns]

        # Combine masks using the AND operator
        combined_mask = functools.reduce(
            operator.and_, online_single_param_masks
        ).fillna(False)

        combined_data = pd.concat(data_list, axis=1)

        return combined_mask, combined_data

    def get_turbines(self):
        """This method parses the CSV data file and identifies all distinct turbine names
        based on the naming convention used in the column names. For each distinct turbine name,
        a new Turbine object is created and added to the `turbines` dictionary.

        Returns:
            dict: A dictionary of Turbine objects, keyed by the turbine name.
        """

        turbines = {}
        for col_name in self.repository.data.columns:
            turbine_name = self._turbine_name_func(col_name)
            # check if this is a new turbine and create a Turbine object if necessary
            if turbine_name not in turbines:
                turbines[turbine_name] = Turbine(
                    turbine_name, data=self.get_subset(turbine_name=turbine_name)
                )

        return turbines

    def get_component_types(self):
        """Returns a list of normalized component type names,
            based on the column names in the input data file.

            Note: Only returns a type if the column type parsed from the file is in the
            constants.KEY_TO_NAME dict.

        Returns:
            A list of normalized component type names.


        """
        component_type_list, tag_suffixes = get_component_types(
            component_type_func=self._component_type_func,
            data=None,
            columns=self.repository.get_all_column_names(),
        )

        return component_type_list, tag_suffixes

    def get_turbine_names(self):
        """Returns a list of all turbine names in the wind farm.

        Returns:
            A list of all turbine names in the wind farm.
        """
        return list(self.turbines.keys())

    def get_turbine(self, name):
        """Returns the Turbine object with the specified name.

        Args:
            name: The name of the turbine to retrieve.

        Returns:
            A Turbine object.
        """

        return self.turbines[name]

    def get_flagged_turbines(
        self, start=None, end=None, period="6H", density_thresh=0.9, n_std=1, top_n=100
    ):
        """
        Retrieves the flagged turbines and their severity scores farm wide over all components within a specified time range.

        Args:
            start (Optional): Start date or timestamp of the time range. Defaults to None.
            end (Optional): End date or timestamp of the time range. Defaults to None.
            period (str): Pandas keyword period for data aggregation. Defaults to "6H" (6 hours).
            density_thresh (float): Threshold value for density-based anomaly detection. Defaults to 0.9.
            n_std (int): Number of standard deviations for anomaly detection. Defaults to 1.
            top_n (int): Number of top flagged turbines on the farm to retrieve. Defaults to 100.

        Returns:
            flagged_turbines (pandas.DataFrame): DataFrame containing the flagged turbines and their severity scores.
                Columns: "Turbine" (formatted turbine name with component type) and "Severity".
            daily_severities (pandas.DataFrame): DataFrame containing the daily severity scores for all components.
                Columns represent the component names, and rows represent daily periods.

        Raises:
            None.

        """
        all_turbines = []
        severity_scores = []

        for component in self.components:
            if component not in DO_NOT_CALCULATE_SEVERITY:
                print(f"[WindFarm 496] processing severity {self.name}, {component}...")
                this_flagged_turbines, daily_severity_scores = self.components[
                    component
                ].get_flagged_turbines(n_std=n_std, top_n=top_n, start=start, end=end)
                all_turbines.append(this_flagged_turbines)
                severity_scores.append(daily_severity_scores)

        daily_severities = pd.concat(severity_scores, axis=1)
        flagged_turbines = pd.concat(all_turbines).sort_values(ascending=False)
        flagged_turbines = pd.DataFrame(
            list(flagged_turbines.items()), columns=["Turbine", "Severity"]
        )

        flagged_turbines["Turbine"] = flagged_turbines["Turbine"].apply(
            lambda x: f"{get_turbine(x)}-{get_component_type(x)}"
        )

        return flagged_turbines, daily_severities


class FarmComponent:
    """

    A class representing a single component of a farm, such as a main bearing, hs bearing, generator temp, etc.
    The FarmComponent class can be used to report on various aspects of a component, such as the average temperature,
    the standard deviation, and more.

    Attributes:
        name (str): The name of the farm component.
        data (pandas.DataFrame): A pandas dataframe containing time-series data for this farm component.
        freq (str): The time frequency of the data, in pandas offset string format.


    """

    def __init__(
        self, name, project, technology=None, data=None, online_map=None, freq="10T"
    ):
        """
        A class representing a single component of a farm, such as a main bearing, hs bearing, generator temp....
        "FarmComponent" doesn't need to be an actual physical component but rather it can be anything that we want to
        report on. i.e. average main bearing temps, std of farm hs bearing temps

        Args:
            name (str): The name of the farm component.
            technology (str): Model of turbine. can
            data (pandas.DataFrame): A pandas dataframe containing time-series data for this farm component.
            freq (str, optional): The time frequency of the data, in pandas offset string format.
                  Defaults to '10T'.
        """

        self.name = name
        self.project = project
        self.technology = technology
        self.data = data
        self._online_map = online_map
        self._resampled_online_map = None
        self._freq = freq
        self._gradient_parameters = None
        self._gradient_filtered_data = None
        self._gradient_filtered_stats = None
        self._range_filter_parameters = None
        self._range_filtered_data = None
        self._range_filtered_stats = None
        self._online_filtered_data = None
        self._online_filtered_stats = None
        self._clean_data = None
        self._z_scores = None
        self._window_severity_score = None
        self._daily_mean = None
        self._park_mean = None
        self._park_std = None

        for agg_type in AggTypes:
            # set empty empty private property for each agg type
            # AggTypes are mean, median, max, min,std
            setattr(self, f"_{agg_type.value}", None)
            setattr(self, f"_{agg_type.value}_gradient", None)
            setattr(self, f"_{agg_type.value}_gradient_stats", None)
            setattr(self, f"_{agg_type.value}_range", None)
            setattr(self, f"_{agg_type.value}_range_stats", None)
            setattr(self, f"_{agg_type.value}_online", None)
            setattr(self, f"_{agg_type.value}_online_stats", None)

            # public properties
            setattr(
                FarmComponent,
                agg_type.value,
                property(
                    lambda self, agg_type_str=agg_type.value: self._get_property_value(
                        agg_type_str
                    )
                ),
            )

    @property
    def gradient_filter_parameters(self):
        if self._gradient_parameters is None:
            if self.project in PROJECT_SUBSETS:
                self._gradient_parameters = GRADIENT_FILTER_PARAMETERS[self.project][
                    self.technology
                ][self.name]
            else:
                # take the value, which is the param dict,  of the first key value pair
                self._gradient_parameters = list(
                    GRADIENT_FILTER_PARAMETERS[self.project].items()
                )[0][1][self.name]
        return self._gradient_parameters

    @property
    def gradient_filtered_stats(self):
        if self._gradient_filtered_stats is None:
            (
                self._gradient_filtered_data,
                self._gradient_filtered_stats,
            ) = gradient_filter(self.data, **self.gradient_filter_parameters)
        return self._gradient_filtered_stats

    @property
    def gradient_filtered_data(self):
        """
        returns a mask
        """
        if self._gradient_filtered_data is None:
            (
                self._gradient_filtered_data,
                self._gradient_filtered_stats,
            ) = gradient_filter(self.data.copy(), **self.gradient_filter_parameters)
        return self._gradient_filtered_data > -1000

    @property
    def range_filter_parameters(self):
        if self._range_filter_parameters is None:
            if self.project in PROJECT_SUBSETS:
                self._range_filter_parameters = RANGE_FILTER_PARAMETERS[self.project][
                    self.technology
                ][self.name]
            else:
                # take the value, which is the param dict,  of the first key value pair
                self._range_filter_parameters = list(
                    RANGE_FILTER_PARAMETERS[self.project].items()
                )[0][1][self.name]

        return self._range_filter_parameters

    @property
    def range_filtered_stats(self):
        if self._range_filtered_stats is None:
            self._range_filtered_data, self._range_filtered_stats = range_filter(
                self.data, **self.range_filter_parameters
            )
        return self._range_filtered_stats

    @property
    def range_filtered_data(self):
        if self._range_filtered_data is None:
            self._range_filtered_data, self._range_filtered_stats = range_filter(
                self.data.copy(), **self.range_filter_parameters
            )
        return self._range_filtered_data > -1000

    @property
    def online_filtered_data(self):
        """
        returns a dataframe of numeric values
        """
        this_map = pd.DataFrame()

        if self._online_map is not None:
            # if the frequencies are not equl, resample the map
            # so it can be applied correctly to the data
            map_freq = pd.infer_freq(self._online_map.index)
            # if any(x in self.name for x in ['Dir','Yaw']):

            if map_freq != self._freq:
                self._online_map = self._online_map.resample(self._freq).ffill()

            if self._online_map.shape[1] != self.data.shape[1]:
                # adjust the online map shape (wrt columns)to match the data
                # for each column in the data match the turbine in the
                # map and repeat it to form a dataframe of the same shape as data
                for col in self.data.columns:
                    this_turbine = get_turbine(col)
                    if this_turbine not in self._online_map:
                        continue
                    if isinstance(self._online_map[this_turbine], pd.Series):
                        this_turbine_map = self._online_map[this_turbine].to_frame()
                    else:
                        this_turbine_map = self._online_map[this_turbine]
                    if len(this_map) == 0:
                        this_map = this_turbine_map
                    else:
                        this_map = pd.concat(
                            [this_map, this_turbine_map],
                            axis=1,
                        )
            else:
                this_map = self._online_map
                this_map.columns = self.data.columns
            self._online_filtered_data = self.data.copy()[this_map].fillna(-9992)

            return self._online_filtered_data
        else:
            print("WindFarm 532: self._online_map is None, returning original dataset")

            return self.data

    @property
    def online_filtered_data_map(self):
        """
        returns a mask with the same shape as the FarmComponent Data
        """

        return self.online_filtered_data > -1000

    @property
    def clean_data(self):
        """Fully cleaned data subject to the combined effect of each cleaning method
        in the order specified by the Utils.Constants.General.CLEANING_ORDER constant.
        """
        datamaps = [getattr(self, clean_type) for clean_type in CLEANING_ORDER]

        combined_clean_mask = functools.reduce(operator.and_, datamaps)

        clean_data = self.data.where(combined_clean_mask, other=-9999)

        return clean_data

    @property
    def clean_data_map(self):
        datamaps = [getattr(self, clean_type) for clean_type in CLEANING_ORDER]
        datamaps.insert(0, self.data > -1000)
        combined_clean_mask = functools.reduce(operator.and_, datamaps)

        return combined_clean_mask

    @property
    def daily_mean(self):
        if self._daily_mean is None:
            component_daily_mean = self._get_daily_mean()
            component_daily_mean.columns = [
                f"{x}_mean" for x in component_daily_mean.columns
            ]
            self._daily_mean = component_daily_mean
        return self._daily_mean

    def _get_daily_mean(self):
        """Calculates the daily mean of this component's values.

        Returns:
            (pandas.DtaFrame): Includes a column for each turbine across this component type. The index will be
                a daily date.

        """
        this_clean_data = self.clean_data.copy()
        this_clean_data = this_clean_data[this_clean_data > -1000]
        daily_mean = this_clean_data.resample("D").mean()

        return daily_mean

    def _get_property_value(self, agg_type_str):
        if getattr(self, f"_{agg_type_str}") is None:
            setattr(self, f"_{agg_type_str}", self.statistic(agg_type_str))

        return getattr(self, f"_{agg_type_str}")

    def calculate_data_removal_stats(self, *datamaps, data_stage_names=CLEANING_ORDER):
        """
        Produces a DataFrame with column names as the index and stage/cleaning method as columns for all cleaning methods.
        This is different than calling for stats from each moethod individual as this method
        takes into account the order of cleaning and does not double count points removed by methods passed
        earlier in the datamaps list.
        The first datamap is always what's missing and not in the raw data.

        Args:
            *datamaps (pd.DataFrame): Variable number of cleaned datamaps.
            data_stage_names (list[str], optional): List of stage/cleaning method names. Defaults to CLEANING_ORDER.

        Returns:
            pd.DataFrame: DataFrame with column names as index and stage/cleaning method as columns, showing percent and
            count of removed data for each cleaning method.

        """
        if not datamaps:
            datamaps = [getattr(self, clean_type) for clean_type in CLEANING_ORDER]

        result = pd.DataFrame(index=self.data.columns)

        raw_data_map = self.data > -1000
        datamaps = list(datamaps)
        datamaps.insert(0, raw_data_map)

        cumulative_datamap = ~raw_data_map
        # Calculate percent and count of removed data for the first datamap
        percent_removed = ((1 - raw_data_map).mean()) * 100
        count_removed = (1 - raw_data_map).sum()
        result["Raw - Percent Missing"] = percent_removed
        result["Raw - Count Missing"] = count_removed

        # Calculate percent and count of removed data for subsequent datamaps
        for i, datamap in enumerate(datamaps[1:], start=1):
            datamap.columns = self.data.columns
            previous_datamap = datamaps[i - 1]

            # assumed here to not go from false to true only true to false
            new_removed_data = (~datamap) & previous_datamap & ~cumulative_datamap

            cumulative_datamap = cumulative_datamap | new_removed_data
            count_removed = new_removed_data.sum()
            result[f"{data_stage_names[i-1]} - Percent Removed"] = (
                new_removed_data.mean()
            ) * 100
            result[f"{data_stage_names[i-1]} - Count Removed"] = count_removed

        final_data_map = cumulative_datamap
        percent_removed = (final_data_map.mean()) * 100
        count_removed = (final_data_map).sum()
        result["Full Clean - Percent Missing"] = percent_removed
        result["Full Clean - Count Missing"] = count_removed

        return result

    def get_flagged_turbines(
        self,
        start=None,
        end=None,
        period="6H",
        density_thresh=0.9,
        n_std=1,
        top_n=10,
        daily_threshold=0.9,
    ):
        """Returns the top n turbines based on their severity scores calculated over a given time period.

        Args:
            start (str or datetime, optional): Start time for severity score calculation. If not specified, the start time
                is set to the first time in the input data. Defaults to None.
            end (str or datetime, optional): End time for severity score calculation. If not specified, the end time
                is set to the last time in the input data. Defaults to None.
            period (str, optional): Time period over which to calculate severity scores, in pandas offset string format.
                Defaults to '6H'.
            density_thresh (float, optional): The proportion of non-NaN values in a window that must exceed the threshold
                value to be included in the window severity calculation. Defaults to 0.9.
            n_std (float, optional): The number of standard deviations from the mean to set as the threshold for flagging
                data points. Defaults to 1.
            top_n (int, optional): Number of turbines to return in the output, sorted by total severity score. Defaults to 10.
            daily_threshold (float): below this proportion of valid to possible records a given turbine's
                data can be thrown out for the day. Each day, component typ and typed turbine is evaluated
                separately.
        Returns:
            tuple (pandas.Series, pandas.DataFrame): A pandas series containing the top n turbines, sorted by total severity score
                and a pandas.DataFrame containing daily severity scores for each turbine and component type
        """

        daily_severity_scores = self.get_severity_scores(
            period=period,
            density_thresh=density_thresh,
            n_std=n_std,
            daily_threshold=daily_threshold,
        )
        if start is None:
            start = daily_severity_scores.index[0]
        if end is None:
            end = daily_severity_scores.index[-1]

        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        # Get the total severity score for each turbine
        turbine_scores = daily_severity_scores.loc[start:end].sum()

        # Get the top n turbines by total severity score
        top_turbines = turbine_scores.nlargest(top_n)

        return top_turbines, daily_severity_scores

    @functools.lru_cache(maxsize=None)
    def get_severity_scores(
        self, period="6H", density_thresh=0.9, n_std=1, daily_threshold=0.9
    ):
        """
        Calculates severity scores for each turbine-component in the input data over time, based on the number of
        standard deviations each value is from its row mean.

        Args:
            start (str or datetime, optional): Start time for data selection. Defaults to None.
            end (str or datetime, optional): End time for data selection. Defaults to None.
            period (str, optional): Time period over which to calculate severity scores, in pandas offset string format.
                Defaults to '6H'.
            density_thresh (float, optional): The proportion of non-NaN values in a window that must exceed the threshold
                value to be included in the window severity calculation. Defaults to 0.9.
            n_std (float, optional): The number of standard deviations from the mean to set as the threshold for flagging
                data points. Defaults to 1.

        Returns:
            pandas.DataFrame: A dataframe containing the severity scores for each turbine-component over time.
                The index contains the date and time of each severity score, and the columns correspond to each turbine-component.
        """
        # complete date range --used in multiple places
        all_intervals = pd.date_range(
            start=self.data.index.min(), end=self.data.index.max(), freq=self._freq
        )
        # make sure we have all time stamps
        df = self.clean_data.copy().reindex(all_intervals, fill_value=np.nan)

        # remove full days for each applicable turbine if
        # the recovery for that day is not high enough
        df = df.groupby(df.index.date).apply(
            lambda x: x.apply(
                lambda y: filter_days(y, freq=self._freq, threshold=daily_threshold)
            )
        )

        # Now Calculate row mean and standard deviation
        row_mean = df[df > -1000].mean(axis=1)
        row_std = df[df > -1000].std(axis=1)

        self._park_mean = row_mean
        self._park_std = row_std

        # Calculate z-score dataframe
        z_scores = (
            df[df > -1000] - row_mean.values.reshape(-1, 1)
        ) / row_std.values.reshape(-1, 1)

        self._z_scores = z_scores

        window_severity_frame = calculate_window_severity_with_recovery_threshold(
            z_scores=self._z_scores, period=36, density_thresh=density_thresh
        )

        # Group by day and calculate daily sums
        daily_severity = window_severity_frame.groupby(
            window_severity_frame.index.date
        ).apply(lambda x: x.apply(custom_sum))

        return daily_severity

    def statistic(self, aggregation_type=None):
        """
        Calculates a row-wise statistic for the FarmComponent's data.

        Args:
            freq (str): A string representing the frequency of the time intervals over which to calculate the statistic.
            aggregation_type (AggTypes): An AggTypes enumeration value representing the type of aggregation to perform.

        Returns:
            pd.Series: A pandas Series object representing the computed statistic.

        Raises:
            ValueError: If an invalid aggregation type is specified.
        """
        # Get the corresponding aggregation function for the specified type
        # get gradient params for this component type

        clean_data = self.clean_data[self.clean_data > -1000]

        # apply the specified aggregation type to the clean data frame
        aggregation_func = getattr(clean_data, aggregation_type, None)

        if aggregation_func is None:
            raise ValueError("Invalid aggregation type specified.")

        # Compute the specified aggregation on each row of the data
        statistic = aggregation_func(axis=1).resample(self._freq).mean().ffill()

        # give it a name, used later in chart titles etc..
        statistic.name = aggregation_type

        return statistic


class YawFarmComponent(FarmComponent):
    def __init__(
        self, name, project, technology=None, data=None, online_map=None, freq="10s"
    ):
        super().__init__(name, project, technology, data, online_map, freq)


# ---- Classes handling calculation from base type FarmComponents


class CalculatedFarmComponent:
    """
    handles transforming base data streams into calculated streams
    """

    def __init__(
        self, name, project, technology=None, data=None, freq=None, revenue_dict=None
    ):
        self.name = name
        self.project = project
        self.technology = technology
        self.data = data
        self._park_mean = None
        self._park_std = None
        self._std_filtered = None
        self._window_severity = None
        self._daily_severity = None
        self._lost_energy = None
        self._lost_revenue = None
        self._freq = "10T" if freq is None else freq
        self._efficiency = None
        self._daily_efficiency = None
        self._period = "6H"
        self._revenue_dict = revenue_dict

    @property
    def park_std(self):
        if self._park_std is None:
            self.get_severity_scores()
        return self._park_std

    @property
    def efficiency(self):
        if self._efficiency is None:
            if self.project in PROJECT_SUBSETS:
                nameplate_capacity = TURBINE_TECH[self.project][self.technology]
            else:
                nameplate_capacity = list(TURBINE_TECH[self.project].items())[0][1]

            self._lost_energy, self._efficiency = self.calculate_simple_efficiency(
                df=self.data,
                interval=self._period,
                nameplate_capacity=nameplate_capacity,
            )
        return self._efficiency

    @property
    def lost_energy(self):
        if self._lost_energy is None:
            if self.project in PROJECT_SUBSETS:
                nameplate_capacity = TURBINE_TECH[self.project][self.technology]
            else:
                nameplate_capacity = list(TURBINE_TECH[self.project].items())[0][1]

            self._lost_energy, self._efficiency = self.calculate_simple_efficiency(
                df=self.data,
                interval=self._period,
                nameplate_capacity=nameplate_capacity,
            )
        return self._lost_energy

    @property
    def daily_efficiency(self):
        if self._daily_efficiency is None:
            if len(self.efficiency) == 0:
                return pd.DataFrame()
            else:
                self._daily_efficiency = (
                    self.efficiency[self.efficiency > -1000].resample("1D").mean()
                )
        return self._daily_efficiency

    @property
    def daily_lost_energy(self):
        """
        Sum lost energy for each day. Calculate efficiency by calling
        the efficiency property if lost energy is None. Lost energy
        is only calculated as a by-product of the efficiency calculation.

        Return value is in MWh.
        """
        if len(self.lost_energy) == 0:
            return None
        return_df = self.lost_energy.resample("1D").sum(min_count=1)
        return_df - return_df.groupby(return_df.index.date).sum(min_count=1)
        return return_df

    @property
    def daily_lost_revenue(self):
        """
        Use daily lost energy to map daily lost revenue.

        Note: When hourly revenue numbers become available
            the hourly efficiency needs to passed to the function below
            instead of the daily lost energy. A new Hourly lost
            energy property/function will need to be created at that point
            and substituted here. Now, since the revenue numbers are the same throughout the month
            we are saving overhead.
        """
        if self._lost_revenue is None:
            if self.daily_lost_energy is None:
                return pd.DataFrame()
            else:
                self._lost_revenue = map_mwh_to_revenue(
                    df=self.daily_lost_energy, revenue_dict=self._revenue_dict
                )
        return self._lost_revenue

    def get_flagged_turbines(
        self, start=None, end=None, period="6H", density_thresh=0, n_std=0, top_n=10
    ):
        """Returns the top n turbines based on their severity scores calculated over a given time period.

        Args:
            start (str or datetime, optional): Start time for severity score calculation. If not specified, the start time
                is set to the first time in the input data. Defaults to None.
            end (str or datetime, optional): End time for severity score calculation. If not specified, the end time
                is set to the last time in the input data. Defaults to None.
            period (str, optional): Time period over which to calculate severity scores, in pandas offset string format.
                Defaults to '6H'.
            density_thresh (float, optional): The proportion of non-NaN values in a window that must exceed the threshold
                value to be included in the window severity calculation. Defaults to 0.9.
            n_std (float, optional): The number of standard deviations from the mean to set as the threshold for flagging
                data points. Defaults to 1.
            top_n (int, optional): Number of turbines to return in the output, sorted by total severity score. Defaults to 10.

        Returns:
            pandas.Series: A pandas series containing the top n turbines, sorted by total severity score.
        """

        daily_severity_scores = self.get_severity_scores(
            period=period, density_thresh=density_thresh, n_std=n_std
        )
        if daily_severity_scores is None or len(daily_severity_scores) == 0:
            return None, None
        if start is None:
            start = daily_severity_scores.index[0]
        if end is None:
            end = daily_severity_scores.index[-1]

        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        # Get the total severity score for each turbine
        turbine_scores = daily_severity_scores.loc[start:end].sum()

        # Get the top n turbines by total severity score
        top_turbines = turbine_scores.nlargest(top_n)

        return top_turbines, daily_severity_scores

    def calculate_simple_efficiency(
        self, df, interval, nameplate_capacity, daily_threshold=None
    ):
        """Calculates lost energy and simple efficiency for each turbine.

        Args:
            df: A pandas DataFrame with a DateTimeIndex and columns for active power and expected power for each turbine.
                Active power columns are named like 'WAK-T001-KW' and expected power columns are named like
                'WAK-T001-EXPCTD-KW-CALC'.
            interval: The time interval as a string to resample the data to (e.g., '5min', '1H', etc.). This dictates
                the rolling window over which the expected and active power are summed to calculate the
                rolling efficiency.
            nameplate_capacity: The nameplate capacity of each turbine in Kilowatts. This is used to scale
                the active power before calculating efficiency.

        Returns:
            tuple: A tuple containing two pandas DataFrames:
                - lost_energy_frame: A DataFrame with one column per turbine, named like 'WAK-T001-LOST-ENERGY',
                  representing the lost energy calculated as the difference between active power and
                  expected power.
                - efficiency_frame: A DataFrame with one column per turbine, named like 'WAK-T001-EFFICIENCY',
                  containing the calculated efficiency values.

        Note:
            - The function assumes a 10-minute frequency for the input data.
            - The rolling window for the efficiency calculation is hard-coded to 24 hours with a 90% data
              recovery threshold.
            - Efficiency values are calculated on resampled mean data over the given interval.
            - The returned efficiency values are filled with -9999 where the calculation is not possible due
              to insufficient data.
        """
        # Find all the turbine names in the dataframe

        if daily_threshold is None:
            daily_threshold = 0.9
        turbines = set([get_turbine(col) for col in df.columns])

        active_power_type_str = component_type_map(
            ComponentTypes.ACTIVE_POWER.value, rtn_property_str=False
        )[0]
        expected_power_type_str = component_type_map(
            ComponentTypes.EXPECTED_POWER.value, rtn_property_str=False
        )[0]

        efficiency_frame = pd.DataFrame()
        lost_energy_frame = pd.DataFrame()

        # Loop through each turbine and calculate its efficiency
        for turbine in turbines:
            # Get the columns for this turbine's active power and expected power
            active_power_cols = [
                col
                for col in df.columns
                if turbine in col and active_power_type_str == get_component_type(col)
            ]
            expected_power_cols = [
                col
                for col in df.columns
                if turbine in col and expected_power_type_str in col
            ]

            # complete date range --used in multiple places
            all_intervals = pd.date_range(
                start=df.index.min(), end=df.index.max(), freq=self._freq
            )

            # drop missing records to create a coincident data set
            df_input = df.loc[:, active_power_cols + expected_power_cols]

            df_input_filtered = df_input[df_input > -1000].dropna()

            if len(df_input_filtered) == 0:
                continue

            df_input_filtered = df_input_filtered.groupby(
                df_input_filtered.index.date
            ).apply(lambda x: filter_days(x, threshold=daily_threshold))

            df_input_filtered = df_input_filtered.reset_index(level="DateTime")
            df_input_filtered = df_input_filtered.set_index("DateTime")
            df_input_filtered = df_input_filtered.reindex(
                all_intervals, fill_value=np.nan
            )

            # Calculate the lost energy and add it to the dataframe
            # This can be non zero when efficiency is missing because
            # efficiency is calculated on a 24 hour rolling sum subject to
            # a 90% recovery threshold
            lost_energy_frame[turbine + "-LOST-ENERGY"] = (
                df_input_filtered[active_power_cols].iloc[:, 0]
                - df_input_filtered[expected_power_cols].iloc[:, 0]
            ) / 6000

            # The filter above handles the recovery of the cleaned data
            # removing any days that have low recovery. The filter below handles
            # the recovery of the rolling window which is only applied to
            # the efficiency calculation because of the potential noisiness of its values
            rolling_window = 144
            min_valid_count = int(
                0.9 * rolling_window
            )  # at least 90% of the values must be valid

            # Calculate the rolling sums for the numerator and denominator
            numerator_rolling_sum = (
                df_input_filtered[active_power_cols]
                .iloc[:, 0]
                .rolling(rolling_window, min_periods=min_valid_count)
                .sum()
            )
            denominator_rolling_sum = (
                df_input_filtered[expected_power_cols]
                .iloc[:, 0]
                .rolling(rolling_window, min_periods=min_valid_count)
                .sum()
            )
            df_input_filtered["ap_rolling"] = numerator_rolling_sum
            df_input_filtered["ep_rolling"] = denominator_rolling_sum

            # Handle cases where the denominator is zero to avoid dividing by zero
            mask = denominator_rolling_sum == 0
            denominator_rolling_sum[
                mask
            ] = 1  # replace zeros with ones; this won't affect the result where the numerator is also zero

            # Calculate the efficiency using the rolling sums and update the dataframe
            efficiency_frame[turbine + "-EFFICIENCY"] = (
                numerator_rolling_sum / denominator_rolling_sum
            ).fillna(-9999)

        return lost_energy_frame, efficiency_frame

    def aggregate_efficiency(self, df, time_interval="1D"):
        """
        This function takes the severity already calculated at the interval needed by
        severity scores and aggregates to the pandas Period ALias string passed in.

        Args:
            df (pandas.DataFrame): the already calculated efficiency data at the interval
                required by get_severity_scores.
            time_interval (pandas Period ALias string): the interval to use when averaging the efficiency data

        """
        if len(df) == 0:
            return None

        # Ensure the dataframe has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The dataframe must have a datetime index.")

        # convert any missings to NaN/None so they are ignored by mean()
        df = df[df > -1000]
        aggregated_df = df.resample(time_interval).sum(min_count=1)

        return aggregated_df

    @functools.lru_cache(maxsize=None)
    def get_severity_scores(self, period="6H", density_thresh=0, n_std=0):
        """
        Calculates severity scores for each turbine-component in the input data over time, based on the number of
        standard deviations each value is from its row mean.

        Args:
            start (str or datetime, optional): Start time for data selection. Defaults to None.
            end (str or datetime, optional): End time for data selection. Defaults to None.
            period (str, optional): Time period over which to calculate severity scores, in pandas offset string format.
                Defaults to '6H'.
            density_thresh (float, optional): The proportion of non-NaN values in a window that must exceed the threshold
                value to be included in the window severity calculation. **** this is not applicable for calculated farm component because
                the metric must be windowed to calculate a meaningful number so its not possible to evaluate how many 10 min values
                did or did not exceed the threshold.
            n_std (float, optional): The number of standard deviations from the mean to set as the threshold for flagging
                data points. Defaults to 1.

        Returns:
            pandas.DataFrame: A dataframe containing the daily severity scores for each turbine-component over time.
                The index contains the date and time of each severity score, and the columns correspond to each turbine-component.
        """

        lost_energy_df = self.lost_energy

        if len(lost_energy_df) == 0:
            return None

        lost_energy_df = lost_energy_df.replace(np.inf, np.nan)
        # Calculate row mean and standard deviation
        row_mean = lost_energy_df[lost_energy_df > -1000].mean(axis=1)
        row_std = lost_energy_df[lost_energy_df > -1000].std(axis=1)

        self._park_mean = row_mean
        self._park_std = row_std

        # Calculate z-score dataframe
        z_scores = (
            lost_energy_df[lost_energy_df > -1000] - row_mean.values.reshape(-1, 1)
        ) / row_std.values.reshape(-1, 1)

        self._z_scores = z_scores

        # Apply mask to keep only z-scores less than n standard deviations or to the left of the mean
        # mask = np.abs(z_scores) > n_std

        # filter values that are not far enough away from the mean to take note of
        # masked_z_scores = z_scores.where(mask, other=0)

        window_severity_frame = calculate_window_severity_with_recovery_threshold(
            z_scores=self._z_scores, period=36, density_thresh=density_thresh  # minutes
        )

        # Group by day and calculate daily sums
        daily_severity = window_severity_frame.groupby(
            window_severity_frame.index.date
        ).apply(lambda x: x.apply(custom_sum))

        return daily_severity

    def statistic(self, aggregation_type=None):
        """
        Calculates a row-wise statistic for the FarmComponent's data.

        Args:
            freq (str): A string representing the frequency of the time intervals over which to calculate the statistic.
            aggregation_type (AggTypes): An AggTypes enumeration value representing the type of aggregation to perform.

        Returns:
            pd.Series: A pandas Series object representing the computed statistic.

        Raises:
            ValueError: If an invalid aggregation type is specified.
        """
        print(f" wf 236 generating {self.name}.{aggregation_type}")
        # Get the corresponding aggregation function for the specified type
        # get gradient params for this component type

        clean_data = self.clean_data[self.clean_data > -1000]

        # apply the specified aggregation type to the clean data frame
        aggregation_func = getattr(clean_data, aggregation_type, None)

        if aggregation_func is None:
            raise ValueError("Invalid aggregation type specified.")

        # Compute the specified aggregation on each row of the data
        statistic = aggregation_func(axis=1).resample(self._freq).mean().ffill()

        # give it a name, used later in chart titles etc..
        statistic.name = aggregation_type

        # # Convert the row statistic to a Series
        # statistic = pd.Series(row_statistic, index=self.data.index)

        return statistic


class CalculatedYawErrorFarmComponent(CalculatedFarmComponent):
    def __init__(self, name, project, technology=None, data=None, freq="10S"):
        super().__init__(name, project, technology, data, freq)

    def calculate_yaw_error(
        self, df, rolling_window_size=21600, minimum_valid_window_pct=0.3
    ):
        """
        Calculates rolling average yaw error

        Args:
            df: A pandas DataFrame with columns for yaw position and nacelle wind direction for each turbine.
                Yaw Position columns are named like WAK-T001-YAW-DIR and nacelle wind direction columns are named like WAK-T001-WIND-DIR.
            rolling_window_size: The number of points that constitute the the  rolling window
            that the yaw error is averaged over


        Returns:
            A new pandas DataFrame containing one column per turbine, with each column named like WAK-T001-YAW-ERROR.
            Yaw error is calculated as YAW-DIR - WIND-DIR, and subjected to a rolling average window
        """

        if rolling_window_size > len(df):
            raise ValueError(
                "Rolling window size cannot be greater than the length of the dataframe."
            )

        yaw_error_dfs = []

        # Find all the turbine names in the dataframe
        turbines = set([get_turbine(col) for col in df.columns])

        yaw_pos_type_str = component_type_map(
            ComponentTypes.YAW_POSITION.value, rtn_property_str=False
        )[0]
        nac_wind_dir_type_str = component_type_map(
            ComponentTypes.NACELLE_WIND_DIRECTION.value, rtn_property_str=False
        )[0]

        # Loop through each turbine
        for turbine in turbines:
            # Get the columns for this turbine's yaw pos and nac wd
            yaw_pos_cols = [
                col
                for col in df.columns
                if turbine in col
                and yaw_pos_type_str == DEFAULT_PARSE_FUNCS["component_type_func"](col)
            ]
            nac_wind_dir_cols = [
                col
                for col in df.columns
                if turbine in col and nac_wind_dir_type_str in col
            ]

            # Ensure we have one column each for yaw position and nacelle wind direction
            if not does_precompute_yaw_error(self.project):
                if len(yaw_pos_cols) != 1 or len(nac_wind_dir_cols) != 1:
                    raise ValueError(
                        f"Expected one column each for yaw position and nacelle wind direction for turbine {turbine}."
                    )
            else:
                if len(nac_wind_dir_cols) != 1:
                    raise ValueError(
                        f"Expected nacelle wind direction column for turbine {turbine}."
                    )

            # Calculate yaw error
            if does_precompute_yaw_error(project=self.project):
                nac_wind_dir_col = nac_wind_dir_cols[0]
                this_df = df[[nac_wind_dir_col]]
                this_df = this_df[this_df > -1000].dropna()
                yaw_error = this_df[nac_wind_dir_col]
            else:
                yaw_pos_col, nac_wind_dir_col = yaw_pos_cols[0], nac_wind_dir_cols[0]
                this_df = df[[yaw_pos_col, nac_wind_dir_col]]
                this_df = this_df[this_df > -1000].dropna()
                yaw_error = this_df[yaw_pos_col] - this_df[nac_wind_dir_col]

            min_periods = int(rolling_window_size * minimum_valid_window_pct)

            # Compute rolling average yaw error
            rolling_avg_yaw_error = yaw_error.rolling(
                window=rolling_window_size, min_periods=min_periods
            ).median()

            rolling_avg_yaw_error = rolling_avg_yaw_error.groupby(
                rolling_avg_yaw_error.index
            ).agg("first")

            yaw_error_dfs.append(
                rolling_avg_yaw_error.to_frame(name=turbine + "-YAW-ERROR")
            )

        # Concatenate yaw error dataframes for all turbines using an outer join

        result_df = pd.concat(yaw_error_dfs, axis=1, join="outer")

        return result_df

    @functools.lru_cache(maxsize=None)
    def get_severity_scores(self, period="6H", density_thresh=0, n_std=1):
        """
        Calculates severity scores for each turbine-component in the input data over time, based on the number of
        standard deviations each value is from its row mean.

        Args:
            start (str or datetime, optional): Start time for data selection. Defaults to None.
            end (str or datetime, optional): End time for data selection. Defaults to None.
            period (str, optional): Time period over which to calculate severity scores, in pandas offset string format.
                Defaults to '6H'.
            density_thresh (float, optional): The proportion of non-NaN values in a window that must exceed the threshold
                value to be included in the window severity calculation. **** this is not applicable for calculated farm component because
                the metric must be windowed to calculate a meaningful number so its not possible to evaluate how many 10 min values
                did or did not exceed the threshold.
            n_std (float, optional): The number of standard deviations from the mean to set as the threshold for flagging
                data points. Defaults to 1.

        Returns:
            pandas.DataFrame: A dataframe containing the daily severity scores for each turbine-component over time.
                The index contains the date and time of each severity score, and the columns correspond to each turbine-component.
        """

        # get rolling yaw error
        yaw_error_df = self.calculate_yaw_error(
            df=self.data, rolling_window_size=500, minimum_valid_window_pct=0.5
        )

        # Calculate row mean and standard deviation
        row_mean = yaw_error_df[yaw_error_df > -1000].mean(axis=1)
        row_std = yaw_error_df[yaw_error_df > -1000].std(axis=1)

        self._park_mean = row_mean
        self._park_std = row_std

        daily_severity = yaw_error_df.groupby(yaw_error_df.index.date).mean()

        return daily_severity
