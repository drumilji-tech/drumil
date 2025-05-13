"""Making Treemaps for the App."""

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import mode

from Charts.Hovertemplate import (
    append_units_and_round,
    join_hover_pair,
    join_hover_lines,
    gen_hovertemplate,
    TREEMAP_HOVERLABEL,
)
from Utils.Constants import (
    DEFAULT_PARSE_FUNCS,
    KEY_TO_NAME,
)
from Utils.Transformers import (
    get_component_type,
    get_project_component,
    get_project_data,
    get_turbine,
    filter_dates,
    filter_oem,
    join_fault_descriptions,
    remove_acknowledged_values,
)
from Utils.UiConstants import (
    ALL_FAULT_CHART_METRICS,
    FAULT_METRIC_COLUMN_LOOKUP,
    DESCRIPTION_CODE_DELIM,
    PERFORMANCE_METRICS,
    PERFORMANCE_METRICS_LABEL_LOOKUP,
    DEFAULT_CHART_HEIGHT,
)


MARGIN_RIGHT = 120

# See the Plotly Docs for valid colorscale names
# https://plotly.com/python/builtin-colorscales/#builtin-sequential-color-scales
COMP_TEMP_COLORSCALE = "Inferno"
POWER_PERF_COLORSCALE = "Viridis"
FAULT_COLORSCALE = "Inferno"


def remove_columns_with_missing_values(df, start=None, end=None):
    """
    Removes columns from the DataFrame which do not have the same number of valid (non-missing) values
    as the mode of valid value counts within the specified range.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    start (str, optional): The start date as a string. If None, uses the first index.
    end (str, optional): The end date as a string. If None, uses the last index.

    Returns:
    pd.DataFrame: A new DataFrame with the specified columns removed.
    """
    if start is None:
        start = df.index[0]
    if end is None:
        end = df.index[-1]

    # Ensuring start and end are in the index
    if start not in df.index or end not in df.index:
        raise ValueError("Start or end date not found in DataFrame index.")

    # Get the range of dates
    date_range = df.loc[start:end]

    # Count the valid (non-missing) values for each column
    valid_counts = date_range.notnull().sum()

    # Find the mode of the valid counts
    valid_counts_mode = mode(valid_counts)[0][0]

    # Find columns that have a number of valid values equal to the mode
    cols_to_keep = valid_counts[valid_counts == valid_counts_mode].index

    # Keep only those columns
    df_cleaned = df.loc[:, cols_to_keep]

    return df_cleaned


def remove_cols_pp_treemap(treemap_data_simple_eff, sort_by):
    """Remove the columns that are not associated with the given metric.

    Args:
        treemap_data_simple_eff (pd.DataFrame): The simple efficiency
            dataset.
        sort_by (str): One of the values from the `PERFORMANCE_METRICS` constant.
    Returns:
        (pd.DataFrame): A dataframe with the same format as the input.
    """

    # remove the leading dash "-"
    metric = sort_by[1:]

    cols = [
        x
        for x in treemap_data_simple_eff.columns
        if metric == DEFAULT_PARSE_FUNCS["component_type_func"](x)
    ]
    treemap_data_simple_eff = treemap_data_simple_eff[cols]
    return treemap_data_simple_eff


def aggregate_pp_treemap_data(
    treemap_data_simple_eff,
    start=None,
    end=None,
    component_type=None,
    project=None,
    overperforming="underperforming",
    sort_by=None,
):
    """Aggregate the power performance treemap data by sum so we can easily plot it.

    Args:
        treemap_data_simple_eff (pandas.DataFrame): This is the relevant dataset.
        start (datetime|optional): Start date of the time window to consider. If start
            is not provided, the start date will be the first date in the dataset.
        end (datetime|optional): End date of the time window to consider. If
            end is not provided, the end date will be the last date in the dataset.
        component_type (str): String representing a specific component type to
            filter by. If you pass in "All", then all components will be included.
        project (str): String representing a specific project to filter by.
        overperforming (str): One of "underperforming" or "overperforming". This param
            controls the values that get filtered through the dataset, as well as if
            we reverse the colorscale or not.
        sort_by (str): One of the values from the `PERFORMANCE_METRICS` constant.

    Returns:
        (pd.DataFrame): The curated dataset ready for plotting.
    """
    treemap_data_simple_eff = remove_cols_pp_treemap(
        treemap_data_simple_eff=treemap_data_simple_eff, sort_by=sort_by
    )

    if start is None:
        start = treemap_data_simple_eff.index[0]
    if end is None:
        end = treemap_data_simple_eff.index[-1]

    # treemap_data_simple_eff = remove_columns_with_missing_values(treemap_data_simple_eff, start=start, end=end)

    treemap_data = sum_cols_pp_treemap(
        treemap_data_simple_eff=treemap_data_simple_eff,
        sort_by=sort_by,
        overperforming=overperforming,
    )
    treemap_data = treemap_data.round(2)

    treemap_data = treemap_data.to_frame().reset_index()
    treemap_data.columns = ["Turbine", "Severity"]

    treemap_data["TurbineRaw"] = treemap_data["Turbine"]
    treemap_data["Turbine"] = treemap_data["Turbine"].apply(
        lambda x: f"{get_turbine(x)}-{KEY_TO_NAME[get_component_type(x)]}"
    )

    if project != "All":
        treemap_data = get_project_data(treemap_data, project)

    treemap_data = treemap_data.sort_values(by="Severity", ascending=False).head(25)

    return treemap_data


def sum_cols_pp_treemap(
    treemap_data_simple_eff, sort_by, overperforming, start=None, end=None
):
    """Intelligently aggregate the column in question.

    Args:
        See `aggregate_pp_treemap_data` as it contains the same params.

    Returns:
        (pd.DataFrame): The aggregated and summed up dataframe.
    """
    if start is None:
        start = treemap_data_simple_eff.index[0]
    if end is None:
        end = treemap_data_simple_eff.index[-1]

    if sort_by == "-LOST-REVENUE":
        if overperforming == "overperforming":
            treemap_data = treemap_data_simple_eff.loc[
                :, treemap_data_simple_eff.loc[start:end].sum() >= 0
            ]
        else:
            treemap_data = (
                treemap_data_simple_eff.loc[
                    :, treemap_data_simple_eff.loc[start:end].sum() < 0
                ]
                * -1
            )
    elif sort_by == "-LOST-ENERGY":
        if overperforming == "overperforming":
            treemap_data = treemap_data_simple_eff.loc[
                :, treemap_data_simple_eff.loc[start:end].sum() >= 0
            ]
        else:
            treemap_data = (
                treemap_data_simple_eff.loc[
                    :, treemap_data_simple_eff.loc[start:end].sum() < 0
                ]
                * -1
            )

    elif sort_by == "-SEVERITY":
        if overperforming == "overperforming":
            treemap_data = treemap_data_simple_eff.loc[
                :, treemap_data_simple_eff.loc[start:end].sum() > 0
            ].fillna(0)

        else:
            treemap_data = (
                treemap_data_simple_eff.loc[
                    :, treemap_data_simple_eff.loc[start:end].sum() <= 0
                ].fillna(0)
                * -1
            )
    treemap_data = treemap_data.loc[start:end].sum()
    return treemap_data


def generate_power_performance_treemap(
    treemap_data_simple_eff,
    start=None,
    end=None,
    component_type=None,
    project=None,
    overperforming="underperforming",
    acknowledged_pp_turbines=None,
    sort_by="-LOST-REVENUE",
):
    """
    Generates a power performance treemap.

    Args:
        treemap_data_simple_eff (pandas.DataFrame): This is the relevant dataset.
        start (datetime|optional): Start date of the time window to consider. If start
            is not provided, the start date will be the first date in the dataset.
        end (datetime|optional): End date of the time window to consider. If
            end is not provided, the end date will be the last date in the dataset.
        component_type (str): String representing a specific component type to
            filter by. If you pass in "All", then all components will be included.
        project (str): String representing a specific project to filter by.
        overperforming (str): One of "underperforming" or "overperforming". This param
            controls the values that get filtered through the dataset, as well as if
            we reverse the colorscale or not.
        acknowledged_pp_turbines (list|tuple of str): An array of turbine-code
            labels that are not meant to show up in the final treemap.
        sort_by (str): One of the values of `PERFORMANCE_METRICS`. This param controls
            how the data is sorted in the treemap.

    Returns:
        plotly.graph_objects.Figure: A Plotly Treemap visualization.
    """

    def _join_all_metrics(df, start, end, sort_by, overperforming):
        """Ensure the dataset contains all metrics to display in the hover.

        This function adds 3 more columns to the already-sorted input dataset.
        Each of these columns contain the values of each metric for the turbines
        in the original dataset.

        Since the power performance treemap will already be sorted for one metric
        in descending order, we have to search the original `treemap_data_simple_eff`
        dataset to filter on dates, over or underperforming, and then finally
        sum up the values and prepare to join.
        """
        names = df["TurbineName"]
        for other_sort in PERFORMANCE_METRICS:
            simple_eff_copy = treemap_data_simple_eff.copy()
            simple_eff_copy = remove_cols_pp_treemap(
                treemap_data_simple_eff=simple_eff_copy, sort_by=other_sort
            )

            other_cols = ["".join([n, other_sort]) for n in names]
            simple_eff_copy = simple_eff_copy[other_cols]
            simple_eff_copy.columns = simple_eff_copy.columns.str.replace(
                other_sort, ""
            )
            simple_eff_copy = filter_dates(simple_eff_copy, start, end, is_index=True)

            dff_sum = sum_cols_pp_treemap(
                treemap_data_simple_eff=simple_eff_copy,
                sort_by=other_sort,
                overperforming=overperforming,
                start=start,
                end=end,
            )
            simple_eff_copy = pd.DataFrame(dff_sum, columns=[other_sort])
            df = df.join(simple_eff_copy, on="TurbineName")
        return df

    def _gen_pp_treemap_hovertemplate(hover_data):
        """Return the curated hovertemplate."""
        lines = [
            join_hover_pair(
                label="Turbine",
                value=append_units_and_round(var="label", metric="Turbine"),
            )
        ]
        for index, datum in enumerate(hover_data):
            label = PERFORMANCE_METRICS_LABEL_LOOKUP[datum]
            lines.append(
                join_hover_pair(
                    label=label,
                    value=append_units_and_round(
                        var=f"customdata[{index}]", metric=label
                    ),
                )
            )
        hovertemplate = join_hover_lines(lines)
        return hovertemplate

    treemap_data = aggregate_pp_treemap_data(
        treemap_data_simple_eff=treemap_data_simple_eff,
        start=start,
        end=end,
        component_type=component_type,
        project=project,
        overperforming=overperforming,
        sort_by=sort_by,
    )

    max_severity = treemap_data["Severity"].max()

    # reverse the colorscale when toggling between over/under-performing
    if overperforming == "overperforming":
        color_continuous_scale = POWER_PERF_COLORSCALE
    else:
        color_continuous_scale = "".join([POWER_PERF_COLORSCALE, "_r"])

    # remove acknowledged turbines before plotting dataset
    filtered_data = treemap_data
    if len(acknowledged_pp_turbines) > 0:
        regex_turbines = "|".join([f"(?:{n})" for n in acknowledged_pp_turbines])
        filtered_data = treemap_data[~treemap_data.Turbine.str.contains(regex_turbines)]

    filtered_data["TurbineName"] = filtered_data["TurbineRaw"].str.replace(sort_by, "")
    filtered_data = _join_all_metrics(
        df=filtered_data,
        start=start,
        end=end,
        sort_by=sort_by,
        overperforming=overperforming,
    )
    filtered_data.fillna(value=0, inplace=True)

    hover_data = PERFORMANCE_METRICS
    treemap = construct_styled_treemap(
        treemap_data=filtered_data,
        color_continuous_scale=color_continuous_scale,
        max_severity=max_severity,
        hovertemplate=None,
        hoverdata=hover_data,
    )
    hovertemplate = _gen_pp_treemap_hovertemplate(hover_data)
    treemap.update_traces(hovertemplate=hovertemplate)
    return treemap


def generate_comp_temp_treemap(
    treemap_data_from_file, start=None, end=None, component_type=None, project=None
):
    """
    Generates a treemap plot using the data from Wind Farm object.

    Args:
        treemap_data_from_file (pd.DataFrame): This is the relevant dataset.
        start (Optional[datetime]): Start date of the time window to consider. Defaults to None.
        end (Optional[datetime]): End date of the time window to consider. Defaults to None.
        component_type (Optional[str]): String representing a specific component type to filter by.
    Defaults to None.

    Returns:
    plotly.graph_objs._figure.Figure: A treemap plot showing the severity of issues in each turbine/component.
    """
    if start is None:
        start = treemap_data_from_file.index[0]
    if end is None:
        end = treemap_data_from_file.index[-1]

    
    mean_col_map = treemap_data_from_file.columns.str.contains(
        "_mean"
    ) & ~treemap_data_from_file.columns.str.contains(
        "|".join(["-DIR", "-YAW", "-LOST"])
    )

    # print('tm 398', start, end, treemap_data_from_file.loc[start:end, ~mean_col_map].copy())
    treemap_data = treemap_data_from_file.loc[start:end, ~mean_col_map].copy().sum()
    
    treemap_data = treemap_data.to_frame().reset_index()
    treemap_data.columns = ["Turbine", "Severity"]
    treemap_data["TurbineRaw"] = treemap_data["Turbine"]

    

    mean_data = treemap_data_from_file.loc[start:end, mean_col_map].copy().mean()
    mean_data.index = mean_data.index.str.replace("_mean", "")
    mean_data = mean_data.to_frame().reset_index()
    mean_data.columns = ["Turbine", "Mean"]
    
    
    project_component_pairs = mean_data["Turbine"].map(get_project_component)
    project_component_pairs = list(set(project_component_pairs))

    
    park_average_dict = {}
    for pc in project_component_pairs:
        matching_cols = [
            col for col in mean_data["Turbine"] if get_project_component(col) == pc
        ]
        park_average_dict[pc] = mean_data.loc[
            mean_data["Turbine"].isin(matching_cols), "Mean"
        ].mean()


    park_average_series = pd.Series(index=mean_data["Turbine"])

    for col in park_average_series.index:
        pc = get_project_component(col)
        park_average_series[col] = park_average_dict[pc]

    park_avg_data = park_average_series.to_frame().reset_index()
    park_avg_data.columns = ["Turbine", "Park Avg"]

    treemap_data = pd.merge(
        treemap_data, mean_data[["Turbine", "Mean"]], on="Turbine", how="left"
    )

    

    treemap_data = pd.merge(treemap_data, park_avg_data, on="Turbine", how="left")

    treemap_data["Turbine"] = treemap_data["Turbine"].apply(
        lambda x: f"{get_turbine(x)}-{KEY_TO_NAME.get(get_component_type(x),'')}"
    )

    if component_type != "All":
        treemap_data = treemap_data[
            treemap_data["Turbine"].str.contains(component_type)
        ]

    if project != "All":
        treemap_data = get_project_data(treemap_data, project)

    treemap_data = treemap_data.sort_values(by="Severity", ascending=False).head(25)

    # Scale the severity values to range [0, 1]
    max_severity = treemap_data["Severity"].max()

    treemap_data = treemap_data[treemap_data["Severity"] > 0].dropna()
    hovertemplate = (
        "<b>Turbine</b>: %{label}<br>"
        "<b>Mean Temperature</b>: %{customdata[0]:.0f}°C<br>"
        "<b>Park Avg Temperature</b>: %{customdata[1]:.0f}°C"
        "<extra></extra>"
    )
    hoverdata = ["Mean", "Park Avg"]
    treemap = construct_styled_treemap(
        treemap_data=treemap_data,
        color_continuous_scale=COMP_TEMP_COLORSCALE,
        max_severity=max_severity,
        hovertemplate=hovertemplate,
        hoverdata=hoverdata,
    )
    return treemap


def generate_fault_treemap(
    df_metric,
    fault_description_df,
    metric,
    start,
    end,
    project=None,
    is_trip=None,
    oem=None,
    acknowledged_faults=None,
):
    """Plots a Treemap displaying some Fault metric for each Turbine/Fault Code.

    Args:
        df_metric (pandas.DataFrame): A dataframe that has a
            pandas.DateTime index (where each row is for a day),
        fault_description_df (pd.DataFrame): This is the dataset
            that contains the Fault Code descriptions. See the
            `load_fault_description` docstring for more info.
        metric (str): One of "downtime", "lost energy", "lost revenue",
            or "count". This param specifies a type of column available
            in `df_metric`.
        start (Optional[datetime]): Start date of the time window
            to consider. Defaults to None.
        end (Optional[datetime]): End date of the time window to
            consider. Defaults to None.
        project (str): Filters the `data_frame` by a project name.
            An example project name is 'PDK-T005'. If "All", then
            no filtering is done.
        is_trip (bool): Determines if we filter our dataset by trip
            (True) or no trips (False). A "Trip" refers to an internal
            threshold that deems a Turbine in need of someone to
            literally visit the Turbine on the ground.
        oem (str): "OEM" stands for Original Equipment Manufacturer.
            This parameter filters the turbine by the OEM. If "All",
            then no filtering is done.
        acknowledged_faults (list): An array of "Code | Description" strings
            that come from the Fault Acknowledge Component.

    Returns:
        (plotly.express.scatter) A Plotly Treemap plot.
    """

    def make_title(start, end, metric):
        """Generate title for the chart."""
        start_title = start.strftime("%b %d, %Y")
        end_title = end.strftime("%b %d, %Y")
        title = f"Fault Code Analysis {metric} ({start_title} to {end_title})"
        return title

    if project is None:
        project = "All"
    if is_trip is None:
        is_trip = False
    if oem is None:
        oem = "All"

    dff = df_metric
    dff = filter_dates(dff, start_fmt=start, end_fmt=end)
    if project != "All":
        dff = get_project_data(dff, project, filter_columns=False)
    dff = filter_oem(data=dff, oem=oem)

    dff = dff.drop("Date", axis=1)
    dff = dff.rename(columns=FAULT_METRIC_COLUMN_LOOKUP)

    treemap_df = dff.groupby(["Turbine", "FaultCode"]).sum()
    treemap_df = treemap_df.reset_index()

    treemap_df = join_fault_descriptions(
        data_frame=treemap_df,
        fault_description_df=fault_description_df,
        code_colname="FaultCode",
    )

    if isinstance(acknowledged_faults, (list, tuple)):
        desc_array = [
            val.split(DESCRIPTION_CODE_DELIM)[0].strip() for val in acknowledged_faults
        ]
        treemap_df = remove_acknowledged_values(
            df=treemap_df,
            values=desc_array,
            colname="FaultCode",
        )

    treemap_df["Downtime"] = round(treemap_df["Downtime"].div(3600), 1)
    treemap_df[["Lost Energy", "Lost Revenue"]] = treemap_df[
        ["Lost Energy", "Lost Revenue"]
    ].round(2)

    treemap_df["TurbineFaultDescription"] = (
        treemap_df["Turbine"] + "<br>" + treemap_df["Description"]
    )
    treemap_df = (
        treemap_df.sort_values(metric, ascending=False).head(25).drop_duplicates()
    )

    hover_data = ["Turbine", "Description", "FaultCode"] + ALL_FAULT_CHART_METRICS
    hovertemplate = gen_hovertemplate(
        hover_data=hover_data,
        x=None,
        y=None,
    )

    treemap = px.treemap(
        treemap_df,
        names="TurbineFaultDescription",
        parents=["" for _ in range(len(treemap_df))],
        values=metric,
        color_continuous_scale=FAULT_COLORSCALE,
        color=metric,
        template="plotly_dark",
        title=make_title(start, end, metric),
        hover_data=hover_data,
    )
    treemap.update_traces(
        pathbar=dict(visible=False),
        marker_cornerradius=4,
        hovertemplate=hovertemplate,
        insidetextfont_size=20,
        selector=dict(type="treemap"),
    )
    treemap.update_layout(
        height=DEFAULT_CHART_HEIGHT + 100,
        uniformtext=dict(minsize=22),
        margin=dict(
            autoexpand=True,
            r=40,
            l=40,
            b=20,
        ),
    )
    return treemap


def construct_styled_treemap(
    treemap_data,
    color_continuous_scale,
    max_severity,
    hovertemplate=None,
    hoverdata=None,
):
    """Take prepared data and put together a Plotly Treemap.

    Args:
        treemap_data (pd.DataFrame): The curated dataset that
            contains Turbines and Severity data ready for plotting.
        color_continuous_scale (list): A Plotly parameter of the
            `plotly.express.treemap` method. Run `help(px.treemap)`
            to learn more about the Plotly Express treemap parameters.
        max_severity (int): The computed maximum severity coming from the dataset.
        hovertemplate (str, optional): ex--> "Turbine: %{label}<br>Mean Temperature: %{customdata[0]:.0f}°C<br>"
        hoverdata (list of str): column names to use for hover data ordered according to the
            hovertemplate
    Returns:
        (A `plotly.graph_objects.Figure` chart)
    """

    def _split_text(name):
        """Separate the project and component substring by a line break."""
        delim = "-"
        res = name.split(delim)
        project = f"{delim}".join(res[:-1])

        if res[-1] == "Lost_Revenue":
            component = "Revenue Delta"
        elif res[-1] == "Lost_Energy":
            component = "Lost Energy"
        elif res[-1] == "Severity":
            component = "Severity"
        else:
            component = res[-1]

        breaks = "<br>"
        return f"{project}{breaks}{component}"

    turbine_arr = treemap_data["Turbine"].tolist()
    names = [_split_text(name) for name in turbine_arr]
    parents = ["" for _ in turbine_arr]
    values = treemap_data["Severity"].tolist()
    color = treemap_data["Severity"].tolist()

    treemap = px.treemap(
        treemap_data,
        names=names,
        parents=parents,
        values=values,
        color_continuous_scale=color_continuous_scale,
        range_color=[0, max_severity],
        color=color,
        template="plotly_dark",
        hover_data=hoverdata,
    )
    treemap.update_traces(
        pathbar=dict(visible=False),
        marker_cornerradius=4,
        hoverlabel=TREEMAP_HOVERLABEL,
    )
    if hovertemplate is not None:
        treemap.update_traces(
            hovertemplate=hovertemplate,
        )
    treemap.update_layout(
        height=DEFAULT_CHART_HEIGHT,
        uniformtext=dict(minsize=22),
        margin=dict(
            autoexpand=False,
            t=18,
            b=0,
            l=0,
            r=MARGIN_RIGHT,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    treemap.update_traces(insidetextfont_size=20, selector=dict(type="treemap"))
    return treemap
