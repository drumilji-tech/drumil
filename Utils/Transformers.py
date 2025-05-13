"""Functions that Transform, Filter, and Wrangle Loaded Data Structures."""
import datetime
import sys

import numpy as np
import pandas as pd

from Utils.Constants import (
    DEFAULT_PARSE_FUNCS,
    KEY_TO_NAME,
    PROJECT_SUBSETS,
    USE_YAW_ERROR_DIRECTLY,
    OEM_PROJECT_MAPPING,
)
from Utils.Enums import ComponentTypes
from Utils.UiConstants import (
    NULL_FAULT_DESCRIPTION,
    TURBINE_FAULT_DELIM,
)

sys.path.append("..")


def format_date_for_filename(stored_date):
    """Convert a `dcc.Store` date for use in a filename."""
    fmt = "%Y_%m_%d"
    stored_date = pd.to_datetime(stored_date)
    return stored_date.strftime(fmt)


def filter_dates(dff, start_fmt, end_fmt, is_index=None, colname=None):
    """
    Filters the rows of a DataFrame based on a date range.

    dff (pd.DataFrame): The dataframe you want to filter.
    start_fmt (str|pd.DateTime): The start date.
    end_fmt (str|pd.DateTime): The end date.
    is_index (bool): Determines on which column the filtering
        is done. If True, filters on the dataframe's index. If
        False, filters on a column named "Date" which must exist.
    colname (str): The column name to filter on.

    Returns:
        pandas.DataFrame: A filtered DataFrame with rows within
            the specified date range.
    """
    if colname is None:
        colname = "Date"
    if is_index is None:
        is_index = False

    start_fmt = pd.to_datetime(start_fmt)
    end_fmt = pd.to_datetime(end_fmt)

    if is_index:
        dff = dff[dff.index >= start_fmt]
        dff = dff[dff.index <= end_fmt]
    else:
        dff = dff[dff[colname] >= start_fmt]
        dff = dff[dff[colname] <= end_fmt]
    return dff


def filter_trip(dff, is_trip):
    """Filter out columns that either contain "TRIP" or "-NOTRIP"."""
    if is_trip is True:
        dff = dff[dff.columns[dff.columns.str.contains("-TRIP")]]
    else:
        dff = dff[dff.columns[dff.columns.str.contains("-NOTRIP")]]
    return dff


def filter_counts(dff):
    """Filter all COUNT columns, or non COUNT columns."""
    dff = dff[dff.columns[dff.columns.str.contains("-COUNT")]]
    return dff


def filter_oem(data, oem, colname=None):
    """Filter Turbines based on a provided OEM.

    Args:
        data (pd.Series or pd.DataFrame): Either a pandas Series that
            should contain Turbine names, or a pandas DataFrame that
            should contain at least one column with Turbine names.

            If `data` is a DataFrame, this function will use the column
            at `colname` to filter the data.
        oem  (str): The name of the OEM to filter by. Can be either
            one of the keys in `OEM_PROJECT_MAPPING` or "All".
        colname (str): Only applicable if `data` is a pandas DataFrame.
            This is the column name that contains the Turbine names.
            Defaults to "Turbine".
    Returns:
        (pd.Series or pd.DataFrame): Returns the datatype that is
            passed into the function.
    """
    if colname is None:
        colname = "Turbine"

    if oem != "All":
        oem_projects = OEM_PROJECT_MAPPING[oem]
        pat = r"|".join(oem_projects)
        if isinstance(data, pd.Series):
            data = data[data.str.contains(pat=pat, regex=True)]
        elif isinstance(data, pd.DataFrame):
            data = data[data[colname].str.contains(pat=pat, regex=True)]
    return data


def remove_acknowledged_values(df, values, colname=None):
    """Remove provided Fault Codes from a given DataFrame column.

    Args:
        df (pandas.DataFrame): The dataset that must include a column
            that contains Fault Code values.
        values (list): The values that we want to remove rows with.
            If values=['5', '6'] for example, we would remove all
            rows that hold fault events with code 5 or 6.
        colname (str): This is the literal column name in `df` that we
            will filter against. Defaults to "FaultCode" if not provided.
    Returns:
        (pandas.DataFrame): The filtered dataframe.
    """
    if colname is None:
        colname = "FaultCode"

    if np.issubdtype(df[colname].dtype, np.number):
        code_values = [int(float(x)) for x in values]
        df = df[~df[colname].isin(code_values)]
    else:
        raise ValueError(
            "This function is intended to filter on a numeric column. "
            "Please make sure you are passing in a numeric column to "
            "filter on and the values passed can be converted to integers. "
            f"You passed in {colname} which is of type {df[colname].dtype}."
        )
    return df


def join_fault_descriptions(data_frame, fault_description_df, code_colname):
    """Add description column to a dataframe that contains column names.

    Beyond joining a column of descriptions, this function also cooerces
    the original code-containing column you are doing the join on, into
    integers. The reason for this that Fault Codes only make sense as integers
    and not float or strings.

    Args:
        data_frame (pd.DataFrame): Any dataframe that must contain a
            column of Fault Codes.
        fault_description_df (pd.DataFrame): The result of calling
            `load_fault_descriptions`.
        code_colname (str): The name of the column name in `data_frame`
            that currently contains fault codes.
    """
    data_frame["Project"] = data_frame["Turbine"].str.extract(r"(\w+)-")
    data_frame[code_colname] = data_frame[code_colname].astype(int)

    df_merged = pd.merge(
        data_frame,
        fault_description_df,
        left_on=[code_colname, "Project"],
        right_on=["Code", "Project"],
        how="left",
    )
    df_merged.drop("Project", axis=1, inplace=True)
    df_merged.drop("Code", axis=1, inplace=True)
    df_merged.fillna(NULL_FAULT_DESCRIPTION, inplace=True)
    return df_merged


def add_turbine_fault_column(fault_metrics_df):
    """Add a column that combines turbine and fault code.

    Args:
        fault_metrics_df (pandas.DataFrame): A dataset that comes
            from the output of the `load_fault_metrics` helper.
    """
    df = fault_metrics_df
    df["TurbineFaultCode"] = (
        df["Turbine"] + TURBINE_FAULT_DELIM + df["FaultCode"].astype(int).astype(str)
    )
    return df


def combine_dataframes(df1, df2, missing_value=-9999):
    """
    Combine two dataframes with the same shape, where missing values are coded as a specific value.
    If either of the input dataframes has missing data, the combined dataframe will have the
    same missing data.

    Args:
        df1 (pandas.DataFrame): The first dataframe to combine.
        df2 (pandas.DataFrame): The second dataframe to combine.
        missing_value (float): The value that represents missing data.

    Returns:
        pandas.DataFrame: The combined dataframe.
    """
    combined_df = np.where(df1 == missing_value, missing_value, df2)
    return pd.DataFrame(combined_df, index=df1.index, columns=df1.columns)


def get_project_data(data, project, filter_columns=None, column_name=None):
    """
    get the correct columns for the project from the input dataset
    which contains all projects. In the event that the project has subsets
    retrieve the list of turbines for the given subset.

    Args:
        data (pandas.DataFrame): the input data to parse columns from
        project (string): comes fromt he project dropdown on the main UI.
                          can be the pure project 3 letter code like 'BTH' or
                          can also be the name of a project subset
                          like 'BR2|GE_2_3'. If it is a subset this function
                          will handle retreiving the correct turbines for the given subset
        filter_columns (bool): Decides how to filter the input data.
            - If True, we look at the columns and make sure only the
                relevant ones are returned.
            - If False, we look inside the values of the column name
                `column_name` and filter from there.
        column_name (str): Name of the column to filter in. Only has an effect
            if filter_columns is False.
    Returns:
        pandas.DataFrame: this will be a subset of data from that which was passed.

    """
    if filter_columns is None:
        filter_columns = False
    if column_name is None:
        column_name = "Turbine"

    if "|" in project:
        project, technology = project.split("|")
        project = project.strip()
        technology = technology.strip()
        turbines = PROJECT_SUBSETS[project][technology]
        turbine_exp = "|".join(turbines)
    else:
        turbine_exp = project

    if filter_columns is False:
        return data[data[column_name].str.contains(turbine_exp)]
    else:
        return data[data.columns[data.columns.str.contains(turbine_exp)]]


def get_component_types(component_type_func, data=None, columns=None):
    """
    Returns a list of normalized component types parsed from the column names of the given data using the provided component_type_func.

    Args:
    data: pandas.DataFrame - The data containing the columns to parse.
    component_type_func: function - A function that takes a column name as input and returns a string representing a potential component type.
    columns: can alternatively to passing in data, pass in a list of columns. this allows a non dataframe source to use the function

    Returns:
    list - A list of normalized component type names found in the column names of the data. Returns an empty list if no component types were found.

    Raises:
    None.

    Notes:
    - The normalization of the component type names is done using the component_type_map function.
    - If a component type cannot be determined from a column name, it will not be included in the returned list.
    """
    component_type_list = []
    tag_type_suffixes = []
    if data is None and columns is None:
        raise ValueError(
            "Transformers.get_component_types: must provide either a dataframe or a list of columns"
        )
    if data is not None:
        columns = data.columns

    for col_name in columns:
        tag_type_suffix = component_type_func(col_name)

        if tag_type_suffix not in tag_type_suffixes:
            tag_type_suffixes.append(tag_type_suffix)

        component_type = component_type_map(tag_type_suffix)
        if component_type is None:
            continue

        if component_type not in component_type_list:
            component_type_list.append(component_type)

    return component_type_list, tag_type_suffixes


def component_type_map(key_str, rtn_property_str=True):
    """
    Maps a component type string to a human-readable name, or vice versa.

    Args:
        key_str (str): The string to be mapped.
        rtn_property_str (bool): If True, maps the key string to a human-readable name (Value of ComponentTypes.Enum).
                                 If FALSE maps the text passed to the key string.

    Returns:
        str: The mapped string.

    """

    # Define dictionaries to map between keys and property names
    # KEY TO NAME dict can come from a db later

    # TODO
    # when multiple tag strings exist for the same normalized
    # component type property name we will have to find the
    # the key value pair that that only applies to this site

    # Invert the KEY_TO_NAME dictionary
    name_to_KEY = {}
    for key, value in KEY_TO_NAME.items():
        if value not in name_to_KEY:
            name_to_KEY[value] = [key]
        else:
            name_to_KEY[value].append(key)

    # Map the key string to a name, or vice versa
    if rtn_property_str:
        return KEY_TO_NAME.get(key_str, None)
    else:
        return name_to_KEY.get(key_str, None)


def merge_csv_files(path):
    """
    Merge CSV Files

    Opens each CSV file individually, parses dates in the first column, sets the first column as the index,
    and merges all the DataFrames together on their indices.

    Args:
        path (str or list): A string representing a single file path or a list of file paths.

    Returns:
        pandas.DataFrame: Merged DataFrame containing data from all the CSV files.

    Example:
        file_paths = ['file1.csv', 'file2.csv', 'file3.csv']
        result_df = merge_csv_files(file_paths)
    """
    if isinstance(path, str):
        # If path is a single string, convert it to a list for consistency
        path = [path]

    dfs = []  # List to store individual DataFrames

    # Open and process each CSV file
    for file_path in path:
        df = pd.read_csv(
            file_path, parse_dates=[0], index_col=0
        )  # Read CSV and parse dates in the first column
        dfs.append(df)  # Add DataFrame to the list

    merged_df = pd.concat(
        dfs, axis=1, join="outer"
    )  # Merge all DataFrames on their indices

    return merged_df


def normalize_compressed(csv_file=None, data=None, type=None, codes=None):
    """
    Normalizes compressed data from a CSV file or DataFrame.

        Reads a compressed data CSV file or DataFrame and performs the following steps to normalize the data:
        1. Extracts datetime and value columns assuming the columns are laid out as datetime, value_col, datetime, value_col, ...
        2. Parses the datetime column using the dateutil.parser.parse function.
        3. Determines the global minimum and maximum datetime from the data.
        4. Creates a datetime index with regular 10-minute intervals.
        5. Stacks the regular intervals on the irregular intervals from the data.
        6. Fills missing values using forward fill method.
        7. Calculates the duration between consecutive timestamps.
        8. Sets all durations to zero where the code present is not in the specified codes list.
        9. Resamples the data by summing the duration values over 10-minute intervals.
        10. Concatenates all the resampled data into a single DataFrame.

        Args:
            csv_file (str): Path to the compressed data CSV file where the even columns starting with 0 are irregular
                            interval datetime columns and the odd columns are the corresponding values for the given turbine.
            data (pandas.DataFrame): Compressed DataFrame with mixed types.
            type (Utils.Enums.ComponentTypes): The type to extract from the compressed data, e.g., ERR-CODE or TURB-STATE-SCADA.
            codes (list of int): List of integers indicating the codes that indicate the turbine is online from the
                                perspective of the passed-in type.

        Returns:
            pandas.DataFrame: The boolean data as a DataFrame to be applied as a mask, where each value is True if the
            specified codes are present for more than half the time.

            pandas.DataFrame: the actual 10 minute data frame where each column is the turbine and each value is the number of seconds
                              the applicable codes are present

    """

    df = data

    # Read the compressed data CSV file
    if df is None:
        df = pd.read_csv(csv_file)

    # Extract datetime and value columns assumes the columns are laid out datetime, value_col, datetime, value_col, ....
    datetime_columns = df.columns[::2]
    value_columns = df.columns[1::2]

    # Initialize a list to store the resampled data
    resampled_data = []
    formats = ["%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S %p"]
    raveled_frame = pd.Series(
        dtype="datetime64[ns]"
    )  # Initialize an empty series with datetime type

    # Check if conversion is necessary and perform conversion
    for col in datetime_columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            # If the column is already a datetime, add it to raveled_frame
            raveled_frame = raveled_frame.append(df[col])
        else:
            for format in formats:
                try:
                    # Convert and append to raveled_frame
                    converted = pd.to_datetime(df[col], format=format, errors="raise")
                    raveled_frame = raveled_frame.append(converted)
                    break  # Exit the loop if conversion is successful
                except (ValueError, TypeError):
                    print(ValueError, TypeError)
                    continue
    # Extract the global minimum and maximum datetime for the entire dataframe before looping through columns
    global_min_datetime = raveled_frame.dropna().min()
    global_max_datetime = raveled_frame.dropna().max()

    if pd.isna(global_min_datetime) or pd.isna(global_max_datetime):
        print(
            "transformers.normalize_compressed global min or max is none , returning None",
            global_min_datetime,
            global_max_datetime,
        )
        return None, None
    # Create a datetime index with regular 10-minute intervals based
    # on global min and max datetimes
    global_intervals = pd.date_range(
        start=global_min_datetime.floor("10min"),
        end=global_max_datetime.floor("10min"),
        freq="10min",
    )

    # Initialize output_mask and output_df with these global intervals
    output_mask = pd.DataFrame(False, index=global_intervals, columns=value_columns)
    output_df = pd.DataFrame(0, index=global_intervals, columns=value_columns)

    # Iterate over each column pair (datetime and value)
    for datetime_col, value_col in zip(datetime_columns, value_columns):
        # a property name/component type enum value is passed in get the type parsed from the
        # column name from that string
        component_type_str = component_type_map(type, rtn_property_str=False)

        if not component_type_str:
            print(
                f"transformers.normalize_compressed: component type {type} not found."
            )
            continue

        # If component_type_str is a string, convert it to a list with a single element
        if isinstance(component_type_str, str):
            component_type_str = [component_type_str]

        # If none of the strings in component_type_str is found in value_col, continue
        if not any(item in value_col for item in component_type_str):
            # print(
            #     f"Transformers.normalize_compressed {component_type_str} not found in {value_col}. Moving to next column pair.  "
            # )
            continue

        this_data = df.loc[:, [datetime_col, value_col]]
        this_data = this_data.dropna()

        if this_data[datetime_col].isnull().all():
            print(f"No data found in {value_col}")
            continue

        for format in formats:
            try:
                this_data.index = pd.to_datetime(
                    this_data[datetime_col], format=format, errors="raise"
                )
                break
            except:
                print(
                    f"transformers.normalize compressed: {format} doesn't work on {this_data.index}"
                )
                continue

        # Determine the global minimum and maximum datetime
        min_datetime = this_data.index.dropna().min()
        max_datetime = this_data.index.dropna().max()

        # Create a datetime index with regular 10-minute intervals
        intervals = pd.date_range(
            start=min_datetime.floor("10min"),
            end=max_datetime.floor("10min"),
            freq="10min",
        )

        # stack the regular intervals on the irregular intervals from the data
        index_series = pd.concat(
            [pd.Series(intervals), pd.Series(this_data.index.dropna())],
            ignore_index=True,
        ).drop_duplicates()

        # Convert index_series to a DataFrame
        index_frame = index_series.to_frame().rename(columns={0: "timestamp"})

        # Sort the DataFrame by the 'timestamp' column
        index_frame.sort_values("timestamp", inplace=True)

        # Check if this_data has a monotonically increasing index
        if not this_data.index.is_monotonic_increasing:
            this_data = this_data.sort_index()

        # Check for duplicates in both this_data.index and index_frame["timestamp"]
        if this_data.index.duplicated().any():
            print("Warning: 'this_data' has duplicated indices.")

        if index_frame["timestamp"].duplicated().any():
            print("Warning: 'index_frame' has duplicated timestamps.")

        try:
            this_data = this_data.reindex(index_frame["timestamp"], method="ffill")
        except:
            this_data.to_csv(f"../southernoperations/AIID/{value_col}.csv")
            raise ValueError("error line 366 Transformers, while reindexing ")

        this_data["duration"] = this_data.index.to_series().diff().shift(periods=-1)

        # Calculate the sum of the duration column where the value column is of normal code value
        mask = this_data[value_col].isin(codes)

        this_data.loc[~mask, "duration"] = pd.Timedelta("0T")
        this_data_resampled = (
            this_data["duration"].dt.total_seconds().resample("10T").sum()
        )
        this_data_resampled.name = value_col

        # Append the resampled data to the list
        resampled_data.append(this_data_resampled)

    if len(resampled_data) > 0:
        output_df = pd.concat(resampled_data, axis=1)
        output_mask = output_df >= 300
    else:
        output_df = pd.DataFrame(columns=value_columns)
        output_mask = pd.DataFrame(columns=value_columns)

    return output_mask, output_df


def get_projects(data, project_func):
    """
    parses column header of input file and returns a list of projects. used in
    generating project drop dwon options for UI

    Args:
        data (pandas.DataFrame): the avg input dataframe
        project_func (lambda): a function used to extract project name from pi tag column names

    Returns:
        list: a list of unique poriject codes like ['WAK','KAY','GNP'...]
    """
    project_list = ["All"]

    for col in data.columns:
        if not any(x in col for x in ["Unnamed", "Time", "Date", "Datetime"]):
            project = project_func(col)
            if project in PROJECT_SUBSETS:
                for technology, turbines in PROJECT_SUBSETS[project].items():
                    this_project = f"{project} | {technology}"
                    if this_project not in project_list:
                        project_list.append(this_project)
            if project not in project_list:
                project_list.append(project)

    return project_list

def get_subset_from_turbine(project, turbine_name):
    """
        Returns the prject subset name that  a turbine belongs to. 
        If it is not part of a project that has subsets then the project passes through 
        and is returned.  
    
    Args:
        project (string): like 'DHW'. 
        turbine_name(str): like 'DHW-T064'
    Returns:
        string:the name of the project subset 
    """
    subset_dicts = PROJECT_SUBSETS.get(project)
    if subset_dicts is not None:
        for subset_name, subset_list in subset_dicts.items():
            if turbine_name in subset_list:
                return subset_name
    else:
        return project


def get_turbine(input_str):
    """Returns the turbine name from a column /Pi Tag name

    Pi Tag or Column must have dash separators. {Project}-{Turbine}-{Rest of the tag}.

    Args:
        input_str (str): the column name string to be parsed
    """
    return DEFAULT_PARSE_FUNCS["turbine_name_func"](input_str)


def get_component_type(input_str):
    """Returns the component type segments  from a column /Pi Tag name

    Pi Tag or Column must have dash separators. {Project}-{Turbine}-{Rest of the tag}.

    Args:
        input_str (str): the column name string to be parsed
    """
    return DEFAULT_PARSE_FUNCS["component_type_func"](input_str)


def get_project_component(col_name):
    project, component = col_name.split("-")[0], get_component_type(col_name)
    return f"{project}-{component}"


def compute_severity_dataset(treemap_data_from_file, start, end, agg=None):
    """Compute the aggregated severity across all Turbines.

    Args:
        treemap_data_from_file (pd.DataFrame): The result of
            `load_treemap_dataset()`.
        start (pandas Timestamp): The start date to filter
            the entire dataset by.
        end (pandas Timestamp): The end date to filter the
            entire dataset by.
        agg: the aggregation to apply. valid values are pandas agg functions
            "sum","min","max","mean","median"....

    Returns:
        severity_df (pd.DataFrame): A dataframe with columns
            "Turbine" and "Severity". The Turbine name only ever
            appears once in its respective column, and its cor-
            responding Severity score is the sum of all the daily
            severity scores across all components of that Turbine.
    """
    if agg is None:
        agg = "sum"
    if start is None:
        start = treemap_data_from_file.index[0]
    if end is None:
        end = treemap_data_from_file.index[-1]
    cols = [
        x
        for x in treemap_data_from_file.columns
        if not any(y in x for y in ["-KW", "-SEVERITY"])
    ]
    
    
    severity_df = treemap_data_from_file.loc[start:end, cols].agg(agg)
    severity_df = severity_df.to_frame().reset_index()
    severity_df.columns = ["Turbine", "Severity"]

    

    severity_df["Turbine"] = severity_df["Turbine"].apply(get_turbine)
    severity_df = severity_df.groupby("Turbine").agg(agg)
    severity_df.reset_index(inplace=True)

    
    return severity_df


def format_columns(data_frame):
    """Format columns of dataframe into project-component pairs.

    Use the anon functions defined in `DEFAULT_PARSE_FUNCS` to
    process the columns of the input dataframe.

    Args:
        data_frame (pandas.DataFrame): The dataframe to format.

    Returns:
        data_frame (pandas.DataFrame)
    """
    data_frame = data_frame.rename(
        columns={
            x: f"{get_turbine(x)}-{KEY_TO_NAME[get_component_type(x)]}"
            for x in data_frame.keys().tolist()
        },
        inplace=False,
    )
    return data_frame


def filter_treemap_columns(
    data_frame, project=None, turbine=None, component_type=None, do_not_show=None
):
    """Filter the treemap dataframe with a formatted project.

    The input dataframe's columns are meant to indicate project-component
    pairs. They are formatted according to anonymous functions defined in
    `DEFAULT_PARSE_FUNCS`.

    Args:
        data_frame (pandas.DataFrame): The dataset to filter. this is
            the treemap data that comes from load_treemap_dataset().
        project (str): The formatted project name of interest.
        turbine (str): the turbine name to filter by
        component_type (str): The formatted component type of interest.
        do_not_show (str): the component types that are in the input file but to not show

    Returns:
        (pandas.DataFrame): A filtered DataFrame.

    """
    if do_not_show is None:
        do_not_show = [
            "Lost_Revenue",
            "Lost_Energy",
            "Efficiency",
            "Simple_Efficiency",
            "Yaw_Error",
        ]

    find_turbine_func = lambda x: x.split("-")[1]
    find_project_func = lambda x: x.split("-")[0]
    find_component_type_func = lambda x: x.split("-")[2]

    dff = data_frame
    columns = [x for x in dff.columns if not any(y in x for y in do_not_show)]

    if project is not None:
        columns = [x for x in columns if find_project_func(x) == project]
    if turbine is not None:
        columns = [x for x in columns if find_turbine_func(x) == turbine]
    if component_type is not None:
        columns = [x for x in columns if find_component_type_func(x) == component_type]

    output = dff[columns]

    return output


def filter_lost_energy(treemap_data_from_file, project):
    """Return a dataset with columns pertaining to lost energy.

    Args:
        treemap_data_from_file (pandas.DataFrame): This param
            comes from the `load_treemap_dataset()` function.
    Returns:
        (lost_energy_df): The filtered DataFrame.
    """
    lost_energy_df = format_columns(treemap_data_from_file)
    lost_energy_df = lost_energy_df.filter(
        regex=f".+-{ComponentTypes.LOST_ENERGY.value}",
        axis=1,
    )
    lost_energy_df = lost_energy_df.loc[
        :, lost_energy_df.columns.str.startswith(project)
    ]
    return lost_energy_df


def filter_mean_values(treemap_data_from_file, project=None, component_type=None):
    """Return a dataset with the mean values.

    Args:
        treemap_data_from_file (pandas.DataFrame): This param
            comes from the `load_treemap_dataset()` function.
    Returns:
        (mean_df): The filtered DataFrame.
    """

    mean_frame = treemap_data_from_file.loc[
        :, treemap_data_from_file.columns.str.endswith("_mean")
    ].copy()

    
    mean_frame.columns = [
        x.replace("_mean", "") for x in mean_frame.columns if "_mean" in x
    ]
    mean_df = format_columns(mean_frame)
    if project is not None:
        project_col_map = mean_df.columns.str.contains(project)
        col_map = project_col_map
    if component_type is not None:
        comp_type_col_map = mean_df.columns.str.contains(component_type)
        col_map = comp_type_col_map
    if (project is not None) and (component_type is not None):
        col_map = project_col_map & comp_type_col_map

    mean_df = mean_df.loc[:, col_map].sort_index(axis=1)

    return mean_df


def MWh_csv_to_dict(file_path):
    """
    Converts a MWh by project CSV file into a nested dictionary structure where the leaf values are dollar amounts.

    The CSV should have a 'Datetime' column, which is split into day and hour,
    and then the remaining columns are treated as projects.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        (dict): A nested dictionary with the structure:
        {project_name: {day: {hour: revenue value to multiply by}}}

    Example:
    Input CSV:

    Datetime           BR2     BTH
    01/01/2021 00:00    10     20
    01/01/2021 01:00    15     25

    Output Dictionary:
    {
        "BR2": {
            "01/01/2021": {
                "00": 10,
                "01": 15
            }
        },
        "BTH": {
            "01/01/2021": {
                "00": 20,
                "01": 25
            }
        }
    }
    """
    df = pd.read_csv(file_path, parse_dates=[0], index_col=[0])

    # Convert the 'Datetime' column into separate 'Day' and 'Hour' columns
    # Convert date to string format
    df["Day"] = df.index.date.astype(str)
    df["Hour"] = df.index.hour

    # Reshaping the DataFrame using melt
    # Modify to melt all columns except 'Day' and 'Hour'
    df_melted = df.melt(id_vars=["Day", "Hour"], var_name="Project", value_name="Value")

    df_pivot = df_melted.pivot_table(
        index=["Project", "Day"], columns="Hour", values="Value", aggfunc="first"
    )

    result = (
        df_pivot.groupby(level=0)
        .apply(lambda x: x.droplevel(0).to_dict(orient="index"))
        .to_dict()
    )

    return result


def map_mwh_to_revenue(df, revenue_dict):
    """
    Maps megawatt-hour values in a DataFrame to their corresponding revenue amounts using a provided dictionary.
    There is no project passed in, the project is extracted from the column names of the input lost energy (MWh)
    dataset.

    Args:
        df (pd.DataFrame): DataFrame containing megawatt-hour values.
        revenue_dict (dict): Dictionary with the structure {project_name: {day: {hour: revenue_per_mwh}}}

    Returns:
        pd.DataFrame: DataFrame with the same structure as the input, but with values converted to revenue amounts.
    """
    if df is None:
        return None
    # Extract days and hours for the entire DataFrame
    days = df.index.date.astype(str)
    hours = df.index.hour

    for col in df.columns:
        project = col.split("-")[0]

        revenue_per_mwh = [
            revenue_dict.get(project, {}).get(day, {}).get(hour, 0)
            for day, hour in zip(days, hours)
        ]

        df[col] *= revenue_per_mwh

    return df


def fill_missing_vals_from_ref_column(df, target_col, ref_col):
    """Transfers values from one column to another in cases of 0 or NaN.

    Args:
        df (pd.DataFrame): The input dataframe. Contains both 'target_col' and 'ref_col'.
        target_col (str): The column name to ensure has valid values.
        ref_col (str): The column from which values are copied if 'target_col' has 0 or NaN.

    Returns:
        pd.DataFrame: The modified dataframe with updated values in 'target_col'.
    """
    df_copy = df.copy()
    df_copy.loc[
        (df_copy[target_col].isna()) | (df_copy[target_col] == 0), target_col
    ] = df_copy.loc[(df_copy[target_col].isna()) | (df_copy[target_col] == 0), ref_col]
    return df_copy


def does_precompute_yaw_error(project):
    """Checks if the current project has a precomputed yaw error available.

    Some manufacturers provide a precomputed yaw error, which is the difference
    between the nacelle position and the actual wind direction. Others only
    offer the nacelle position and the absolute wind direction. In the latter
    scenario, we need to compute the yaw error. This function helps to determine
    whether such computation is necessary.

    Args:
        project (str): Name of the project.

    Returns:
        bool: True if the precomputed yaw is available for this manufacturer,
              False otherwise.
    """

    if project in USE_YAW_ERROR_DIRECTLY:
        return True
    return False


def calculate_window_severity_with_recovery_threshold(z_scores, period, density_thresh):
    """
    Calculate the rolling sum of severity scores, considering only windows that meet a specified recovery rate threshold,
    ensuring all timestamps are returned.

    Args:
    - z_scores (pd.DataFrame): DataFrame with masked Z-scores.
    - period (int): The size of the rolling window.
    - density_thresh (float): The threshold for the recovery rate below which the window is disregarded.

    Returns:
    - pd.DataFrame: The rolling sum of severity scores for windows meeting the recovery rate threshold,
                    with all timestamps included.
    """
    # Initialize a DataFrame to store results, with the same index as the original DataFrame and NaN values
    window_severity_frame = pd.DataFrame(index=z_scores.index, columns=z_scores.columns)
    window_severity_frame[:] = np.nan

    # Perform rolling calculations
    rolling_masked_z_scores = z_scores.rolling(period)
    window_counts = rolling_masked_z_scores.count()

    # Ensure that window_counts and period are numeric and compatible for division
    window_counts = z_scores.rolling(period).count().astype(float)
    period_float = float(period)  # Cast period to float to ensure compatibility

    # Calculate recovery rate for each window
    window_proportion = window_counts / period_float
    valid_windows_mask = window_proportion >= density_thresh

    # Update the result DataFrame for windows that meet the threshold
    window_sum = rolling_masked_z_scores.sum(min_count=1)
    window_severity_frame[valid_windows_mask] = window_sum[valid_windows_mask]

    return window_severity_frame


import pandas as pd


def filter_days(group, interval=None, threshold=None, freq=None):
    """Filter groups of 10-minute data for recovery.

    Args:
        group (pandas.DataFrame or pandas.Series): DataFrame or Series containing the data for the recovery rate.
        interval (integer): The number of minutes represented by each time stamp. Defaults to 10.
        threshold (float): This proportion of records must be valid for the frame. Defaults to 0.9.
        freq (str): Pandas time frequency alias.

    Returns:
        pandas.DataFrame or pandas.Series: Returns an empty DataFrame/Series if the recovery is too low or the original DataFrame/Series if it is not.
    """
    if interval is None:
        interval = 10

    if threshold is None:
        threshold = 0.9

    multiplier = 60
    if freq == "H":
        multiplier = 1
    elif freq == "10T":
        multiplier = 60
    elif freq is None:
        freq = "10T"  # Default frequency
    else:
        raise ValueError(
            f"Frequency '{freq}' not recognized. Handling needs to be added."
        )

    intervals_per_day = 24 * multiplier // interval
    valid_records = group.dropna().shape[0]

    # Check recovery rate
    if valid_records / intervals_per_day < threshold:
        # Return an empty structure of the same type as the input
        return group.head(0)
    else:
        return group


def custom_sum(series):
    """Handle missings differently than the native sum

    Args:
        series (pd.Series): a pandas series

    Returns:
        pandas.Series: returns the original series or a blank series depending on recovery
    """
    if series.isna().all():
        return None
    else:
        # need to have 90 % of a days values to not return NA
        return series.sum(min_count=1)
