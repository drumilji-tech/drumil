"""The Functions and Plotters for the Power Curve."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import dash
import pandas as pd
from numbers import Number

from Utils.UiConstants import PARETO_COLORS, DEFAULT_CHART_HEIGHT
from Utils.Constants import PROJECT_SUBSETS
from Utils.Transformers import get_subset_from_turbine


POWER_CURVE_COLORS = {
    "avg": "#EEEEEE",
    "turbine": px.colors.qualitative.D3[0],
    "oem": px.colors.qualitative.D3[1],
    "shaded": "rgba(255,255,255,0.5)",
}
AVG_LINE_STYLE = dict(color=POWER_CURVE_COLORS["avg"], dash="dashdot", width=2)

MAGIC_COLUMN = 36
DEFAULT_NEIGHBOURS_SELECTED = 3
WS_RANGE = [2, 18]


def calculate_weighted_average_power_by_turbine(df_power, df_counts):
    """
    Calculate the weighted average power for each wind speed bin,
    weighted by the counts in each bin.

    Parameters:
    df_power (DataFrame): DataFrame with power values, indexed by Turbine and Day.
    df_counts (DataFrame): DataFrame with count values, indexed by Turbine and Day.

    Returns:
    pd.Series: Weighted average power for each wind speed bin.
    """
    if isinstance(df_power, pd.Series):
        df_power = df_power.to_frame()
    if isinstance(df_counts, pd.Series):
        df_counts = df_counts.to_frame()

    # Assuming 'Turbine' and 'Day' are in the index, if not, set them as the index.
    if "Turbine" in df_power.columns and "Day" in df_power.columns:
        df_power = df_power.set_index(["Turbine", "Day"])
    if "Turbine" in df_counts.columns and "Day" in df_counts.columns:
        df_counts = df_counts.set_index(["Turbine", "Day"])

    # Separate numeric columns from non-numeric columns
    numeric_cols_power = df_power.columns.difference(["Turbine", "Day"])
    numeric_cols_counts = df_counts.columns.difference(["Turbine", "Day"])

    # Convert numeric columns to float, ignoring non-numeric columns
    df_power = df_power[numeric_cols_power].apply(pd.to_numeric, errors="coerce")
    df_counts = df_counts[numeric_cols_counts].apply(pd.to_numeric, errors="coerce")

    # Group by Turbine and sum the power and counts across all days
    sum_counts = df_counts.sum()

    # Calculate the total weighted power for each bin
    weighted_power = df_power.multiply(df_counts)
    total_weighted_power = weighted_power.sum()

    # Calculate the weighted average power for each bin, per turbine
    weighted_avg_power = total_weighted_power.divide(sum_counts)

    # Sort numeric columns only
    sorted_numeric_cols = sorted(numeric_cols_power, key=lambda x: float(x))
    weighted_avg_power = weighted_avg_power[sorted_numeric_cols]

    return weighted_avg_power, sum_counts


def calculate_weighted_average_power_collapse_bins(df_power, df_counts):
    """
    Calculate the weighted average power for each wind speed bin for each turbine,
    weighted by the counts in each bin.

    Parameters:
    df_power (DataFrame): DataFrame with power values, indexed by Turbine and Day.
    df_counts (DataFrame): DataFrame with count values, indexed by Turbine and Day.

    Returns:
    DataFrame: Weighted average power for each wind speed bin for each turbine.
    """
    # Group by integer wind speed bins after transposing df_counts if necessary

    # shift the index forward so our bins are centered, lining up correctly with the ws dist
    df_counts.index = df_counts.index + 0.5

    df_power.index = df_power.index + 0.5

    # take the sum of counts for the two bins covered by each integer
    sum_counts = df_counts.groupby(df_counts.index.astype(int)).sum()

    # Calculate total weighted power
    total_weighted_power = df_power.loc[df_counts.index].multiply(df_counts)
    collapsed_weighted_power = total_weighted_power.groupby(
        total_weighted_power.index.astype(int)
    ).sum()

    # Calculate weighted average power for each bin, per turbine
    weighted_avg_power = collapsed_weighted_power.divide(sum_counts)

    # calculations are complete, reverse the introduced bin offset
    weighted_avg_power.index = weighted_avg_power.index
    # Convert column names to floats
    weighted_avg_power = weighted_avg_power.transpose()
    weighted_avg_power.columns = pd.to_numeric(
        weighted_avg_power.columns, errors="coerce"
    )

    sum_counts.index = sum_counts.index
    sum_counts = sum_counts.transpose()
    sum_counts.columns = pd.to_numeric(sum_counts.columns, errors="coerce")

    return weighted_avg_power, sum_counts


def calculate_AEP(project, power_curve_df, ws_dist_frame=None):
    """calculate the theoretical Annual Energy Production on a given powercurve

    AEP is calculated by taking the sum-product of the power curve with the wind speed distribution.
    The wind speed distribution is binned by integer wind speeds so before we take the dot product
    we first transform the power curve which is in .5 m/s bins to 1 m/s bins.

    Args:
        project (str): 3 letter code representing the project the AEP is being calculated for. The distributions
            used in the AEP calculation are looked up  by project name.
        power_curve_df (pandas.DataFrame): a dataframe conatining the ws bins in the index and the average power
            in the first column.

    Returns:
        float: the Annual Energy Production of the given power curve. A theoretical number standardizing
                the metric as if the given power curve help exactly as is for a year. This allows two power curves to be compared
                in terms of their energy production not just average power values which do not account for the wind speed distribution
    """
    # convert the power curve to a bin resolution matching the ws distribution

    annual_hours = 8760
    # make sure all power curves are calculating on the same bins

    expected_power_output = ws_dist_frame[project].multiply(power_curve_df)

    # Annual energy production for each bin (kWh)
    annual_energy_per_bin = expected_power_output * annual_hours

    # Total AEP (kWh)
    total_annual_energy = annual_energy_per_bin.sum().sum()

    # Convert to MWh
    total_annual_energy_mwh = int(total_annual_energy / 1000)

    return total_annual_energy_mwh


def can_convert_to_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def adjust_indices_to_match(dataframe, target_indices):
    if not target_indices.empty:
        # Determine the type of the target indices (valid_indices in this case)
        target_type = type(target_indices[0])

        if isinstance(dataframe, pd.DataFrame):
            # Transpose if the columns are numeric (wind speed bins) so that the index becomes the
            # wind speed bins
            if all(can_convert_to_float(col) for col in dataframe.columns):
                dataframe = dataframe.transpose()

        # Convert the dataframe index to the same type
        dataframe.index = dataframe.index.map(target_type)

    return dataframe


def find_common_valid_indices(*power_curves):
    """
    Find common indices where all given power curves have non-zero and non-missing values.

    Parameters:
    *power_curves (variable number of pd.Series): Power curves to be compared.

    Returns:
    pd.Index: Common indices with valid values in all power curves.values are strings.
    """
    valid_indices_list = []

    # Process each power curve dataframe
    for curve in power_curves:
        if isinstance(curve, pd.DataFrame):
            # Transpose if the columns are numeric (wind speed bins) so that the index becomes the
            # wind speed bins
            if all(can_convert_to_float(col) for col in curve.columns):
                curve = curve.transpose()
                curve.index = curve.index.astype(float)
                # drop all rows where all values are Missing and return the row where all value are > 0
                valid_indices = curve[curve > 0].dropna(how="all").index
        else:
            if all(can_convert_to_float(idx) for idx in curve.index):
                curve.index = curve.index.astype(float)
            valid_indices = curve[curve > 0].dropna().index

        valid_indices_list.append(valid_indices)

    # Find the common indices across all power curves
    common_valid_indices = valid_indices_list[0]
    for indices in valid_indices_list[1:]:
        common_valid_indices = common_valid_indices.intersection(indices)

    return common_valid_indices


def full_calculate_AEP(
    project,
    power_curve,
    power_curve_distribution,
    ws_dist,
    valid_indices,
    output_debug_frame=False,
):
    """Consolidate power curve preparation and aep calculation in one call

    Args:
        project (str): 3 letter code for project like 'KAY'
        power_curve (pd.Series): name is turbine, index is ws bins
        power_curve_distribution (pd.Series): name is turbine. columns are ws bins
        ws_dist (_type_): index is ws bins. columns are project or subset names
        valid_indices (index): indices to calculate aep over. when being compared to other
            power curves this allows common indices to be passed in so the resulting
            AEP results are comparable.

    Returns:
        float: AEP value in MWh
    """
    power_curve = adjust_indices_to_match(power_curve, valid_indices)
    power_curve_distribution = adjust_indices_to_match(
        power_curve_distribution, valid_indices
    )

    power_curve = power_curve.loc[valid_indices]
    power_curve_distribution = power_curve_distribution.loc[valid_indices]

    (
        collapsed_turbine_power_curve,
        collapsed_turbine_power_curve_counts,
    ) = calculate_weighted_average_power_collapse_bins(
        power_curve, power_curve_distribution
    )

    (
        turbine_power_curve,
        turbine_power_curve_counts,
    ) = calculate_weighted_average_power_by_turbine(
        collapsed_turbine_power_curve, collapsed_turbine_power_curve_counts
    )

    aep = calculate_AEP(
        project=project,
        power_curve_df=turbine_power_curve,
        ws_dist_frame=ws_dist,
    )

    if output_debug_frame:
        return aep, turbine_power_curve, turbine_power_curve_counts
    else:
        return aep, None, None


def gen_wind_speed_bins(power_curve_df, distribution_df):
    """Find the common wind speed bins between power curve datasets."""
    max_ws_bin = 15

    common_bins = set.intersection(
        set(power_curve_df.columns), set(distribution_df.columns)
    )
    ws_bins = sorted(list(common_bins), key=float)
    ws_bins = [b for b in ws_bins if float(b) <= max_ws_bin]
    return ws_bins


def gen_power_curve_traces(
    turbine,
    oem_power_curve,
    filtered_power_curve,
    filtered_distribution,
    ws_distribution,
    ws_bins,
):
    """
    Generates power curve traces for a specific turbine, including
    OEM (Original Equipment Manufacturer) power curve,
    filtered turbine power curve, and park average power
    curve. Also calculates the Annual Energy Production (AEP)
    difference as a percentage between the target turbine and
    the park average, as well as between the target turbine
    and the warranted (OEM) values.

    Args:
        turbine (str): The name of the target turbine.
        oem_power_curve (DataFrame): The OEM power curve data.
        filtered_power_curve (DataFrame): The filtered power curve data for all turbines.
        filtered_distribution (DataFrame): The filtered distribution data for all turbines.
        ws_distribution (DataFrame): The wind speed distribution data.
        ws_bins (list of str): The wind speed bins.

    Returns:
        tuple: A tuple containing:
            - Three Plotly Scatter objects (trace_oem_power_curve, trace_turbine_power_curve,
            trace_park_avg_power_curve) for plotting the respective power curves.
            - A dictionary (aep_data) with keys formatted as "{turbine} - Park Avg"
            and "{turbine} - Warranted", where values are the rounded percent change
            in AEP relative to the park average and OEM values, respectively.

    Note:
        - This function assumes that wind speed distributions and power curves
        are provided in specific formats and may require preprocessing to match expected data structures.
        - The AEP calculations are based on the assumption that the wind
        speed distribution and power curves are aligned in terms of their wind speed bins.

    """

    project = turbine.split("-")[0]
    numeric_ws_bins = [float(x) for x in ws_bins]
    park_avg_name = get_subset_from_turbine(project, turbine)
    if park_avg_name != project:
        park_avg_name = f'{project}_{park_avg_name}'

    full_oem_power_curve = oem_power_curve
    full_oem_power_curve.columns = [project]

    # Make sure we have the turbine in the power curve dataset
    if (turbine not in filtered_power_curve.index) | (
        turbine not in filtered_distribution.index
    ):
        if turbine not in filtered_power_curve.index:
            print(
                f"PowerCurve.gen_power_curve_traces:{turbine} is not found in Power Curves data."
            )
            filtered_power_curve.to_csv('filtered_power_curve.csv')
        else:
            print(
                f"PowerCurve.gen_power_curve_traces:{turbine} is not found in Power Curve Distrirbution data."
            )
        return (None, None, None), None
    (
        turbine_power_curve,
        turbine_power_curve_counts,
    ) = calculate_weighted_average_power_by_turbine(
        filtered_power_curve.loc[turbine], filtered_distribution.loc[turbine]
    )
    full_turbine_power_curve = turbine_power_curve.copy()

    (
        park_avg_power_curve,
        park_avg_power_curve_counts,
    ) = calculate_weighted_average_power_by_turbine(
        filtered_power_curve.loc[f"{park_avg_name}_Park_Average"],
        filtered_distribution.loc[f"{park_avg_name}_Park_Distribution"],
    )
    full_park_avg_power_curve = park_avg_power_curve.copy()

    common_valid_indices = find_common_valid_indices(
        full_park_avg_power_curve, full_oem_power_curve, full_turbine_power_curve
    )

    # if the ws distribution index is integer resolution, convert the power curves to match
    if all(ws_distribution.index % 1 == 0):
        oem_counts = pd.DataFrame(
            np.ones(oem_power_curve.shape),
            index=oem_power_curve.index,
            columns=oem_power_curve.columns,
        )

        #####------DEBUG----#######
        debug = False
        ##########-DEBUG-##########

        oem_AEP, collapsed_oem_power_curve, collapsed_oem_counts = full_calculate_AEP(
            project=project,
            power_curve=oem_power_curve,
            power_curve_distribution=oem_counts,
            ws_dist=ws_distribution,
            valid_indices=common_valid_indices,
            output_debug_frame=debug,
        )

        (
            turbine_AEP,
            collapsed_turbine_power_curve,
            collapsed_turbine_counts,
        ) = full_calculate_AEP(
            project=project,
            power_curve=filtered_power_curve.loc[turbine],
            power_curve_distribution=filtered_distribution.loc[turbine],
            ws_dist=ws_distribution,
            valid_indices=common_valid_indices,
            output_debug_frame=debug,
        )

        (
            park_avg_AEP,
            collapsed_park_power_curve,
            collapsed_park_counts,
        ) = full_calculate_AEP(
            project=project,
            power_curve=filtered_power_curve.loc[f"{park_avg_name}_Park_Average"],
            power_curve_distribution=filtered_distribution.loc[
                f"{park_avg_name}_Park_Distribution"
            ],
            ws_dist=ws_distribution,
            valid_indices=common_valid_indices,
            output_debug_frame=debug,
        )

        ##### DEBUG #######
        if debug:
            pd.concat(
                [
                    collapsed_oem_power_curve,
                    collapsed_oem_counts,
                    collapsed_turbine_power_curve,
                    collapsed_turbine_counts,
                    collapsed_park_power_curve,
                    collapsed_park_counts,
                    ws_distribution,
                ],
                axis=1,
            ).to_csv("southernoperations/AIID/debug_aep.csv")
        ##################
    aep_data = {}

    aep_data[f"{turbine} - Park Avg"] = round(
        ((turbine_AEP - park_avg_AEP) / park_avg_AEP) * 100, 1
    )
    aep_data[f"{turbine} - Warranted"] = round(
        ((turbine_AEP - oem_AEP) / oem_AEP) * 100, 1
    )

    trace_oem_power_curve = go.Scatter(
        x=ws_bins,
        y=full_oem_power_curve.loc[numeric_ws_bins].iloc[:, 0],
        mode="lines",
        name="OEM Power Curve",
        line=dict(color=POWER_CURVE_COLORS["oem"], width=2),
    )

    trace_turbine_power_curve = go.Scatter(
        x=ws_bins,
        y=full_turbine_power_curve.loc[numeric_ws_bins],
        mode="lines",
        name=f"{turbine} Power Curve",
        fill="tonexty",
        fillpattern=dict(shape="x", size=3, solidity=0.2),
        fillcolor=POWER_CURVE_COLORS["shaded"],
        line=dict(color=POWER_CURVE_COLORS["turbine"], width=2),
    )

    trace_park_avg_power_curve = go.Scatter(
        x=ws_bins,
        y=full_park_avg_power_curve.loc[numeric_ws_bins],
        mode="lines",
        name="Park Average Power Curve",
        line=AVG_LINE_STYLE,
    )
    return (
        trace_oem_power_curve,
        trace_turbine_power_curve,
        trace_park_avg_power_curve,
    ), aep_data


def gen_distribution_traces(turbine, filtered_distribution, ws_bins):
    # Fetch the original counts for turbine and park distribution
    if turbine not in filtered_distribution.index:
        return (None, None)
    turbine_original_counts = filtered_distribution.loc[turbine].sum()
    project = turbine.split("-")[0]
    park_avg_name = get_subset_from_turbine(project, turbine)
    if park_avg_name != project:
        park_avg_name = f'{project}_{park_avg_name}'

    park_avg_original_counts = filtered_distribution.loc[
        f"{park_avg_name}_Park_Distribution"
    ].sum()

    # Create and Normalize the Distributions
    turbine_distribution = filtered_distribution.loc[turbine].sum()
    park_avg_distribution = filtered_distribution.loc[
        f"{park_avg_name}_Park_Distribution"
    ].sum()
    turbine_distribution /= turbine_distribution.sum()
    park_avg_distribution /= park_avg_distribution.sum()

    trace_turbine_dist_counts = go.Bar(
        x=ws_bins,
        y=turbine_distribution[ws_bins],
        name=f"{turbine} Distribution",
        marker_color=POWER_CURVE_COLORS["turbine"],
        hovertemplate="%{x}: %{y:.2%}<br>Original Count: "
        + turbine_original_counts[ws_bins].astype(str)
        + "<extra></extra>",
    )
    trace_turbine_dist_counts_and_pct = go.Bar(
        x=ws_bins,
        y=park_avg_distribution[ws_bins],
        name="Park Average Distribution",
        marker_color=POWER_CURVE_COLORS["avg"],
        hovertemplate="%{x}: %{y:.2%}<br>Original Count: "
        + park_avg_original_counts[ws_bins].astype(str)
        + "<extra></extra>",
    )
    return (
        trace_turbine_dist_counts,
        trace_turbine_dist_counts_and_pct,
    )


def gen_level1_subcharts(
    power_curve_df,
    distribution_df,
    ws_distribution,
    turbine,
    start_date,
    end_date,
):
    """Generate the subcharts for the power curve and distribution.

    This chart appears in the expanded space of a treemap cell when that cell is clicked.

    Returns:
        (plotly.graphs_objects.Figure): A Plotly figure.
    """

    project = turbine.split("-")[0]
    ws_bins = gen_wind_speed_bins(
        power_curve_df=power_curve_df,
        distribution_df=distribution_df,
    )

    project = get_subset_from_turbine(project=project,turbine_name=turbine)

    oem_power_curve = (
        power_curve_df.loc[
            (power_curve_df.index.get_level_values("Day") == "All")
            & (power_curve_df.index.get_level_values("Turbine").str.contains(project))
        ]
        .iloc[0, :]
        .to_frame()
    )

    power_curve_df = power_curve_df.loc[
        power_curve_df.index.get_level_values("Day") != "All"
    ]

    # filter data based on the global controls
    start_date = start_date.split("T")[0]
    end_date = end_date.split("T")[0]
    mask = (power_curve_df.index.get_level_values("Day") >= start_date) & (
        power_curve_df.index.get_level_values("Day") <= end_date
    )

    filtered_power_curve = power_curve_df[mask]
    filtered_distribution = distribution_df[mask]

    power_curve_traces, aep_data = gen_power_curve_traces(
        turbine,
        oem_power_curve,
        filtered_power_curve,
        filtered_distribution,
        ws_distribution,
        ws_bins,
    )
    aep_annotations = []
    aep_text = ""
    if aep_data is not None:
        for i, (turbine_name, aep) in enumerate(aep_data.items()):
            text = f"{turbine_name}: {aep:.1f}%"
            aep_text += text
            aep_text += "<br>"

        aep_annotations.append(
            go.layout.Annotation(
                text=aep_text,
                align="right",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=1,
                y=0,
                xanchor="right",
                yanchor="bottom",
                borderwidth=1,
                bgcolor="#171717",
                opacity=1,
                borderpad=6,
            )
        )

    distribution_traces = gen_distribution_traces(
        turbine,
        filtered_distribution,
        ws_bins,
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_yaxes=True,
        subplot_titles=(
            f"Power Curves ({turbine})",
            f"Frequency Distributions ({turbine})",
        ),
        row_heights=[
            0.75,
            0.25,
        ],  # Added this to make the PowerCurve taller than bar chart - FBB 12/23
    )
    if not any(x is None for x in power_curve_traces):
        fig.add_traces(power_curve_traces, 1, 1)
        fig.add_traces(distribution_traces, 2, 1)
    fig.update_layout(
        height=DEFAULT_CHART_HEIGHT,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
        ),
        template="plotly_dark",
        xaxis2_title="Wind Speed (m/s)",
        yaxis_title="Power (kW)",
        yaxis2_title="Normalized Frequency",
        xaxis2=dict(showgrid=True),
        hovermode="x unified",
        hoverlabel_namelength=-1,
    )
    for annotation in aep_annotations:
        fig.add_annotation(annotation, row=1, col=1)

    fig.update_xaxes(range=[3, None], row=1, col=1)
    fig.update_xaxes(matches="x")
    return fig


def gen_peer_to_peer_chart(
    selected_neighbors, selected_target, surrogation_strategies, filtered_power_curves
):
    """A plot of one turbine's power curve, and its neighbours.

    Args:
        selected_neighbors (list): A list of selected neighbors.
            These will manifest as separate curves in the plot.
        selected_target (str): The target turbine. By clicking on
            one of the tiles of the Power Performance Treemap in
            the UI, of which each tile corresponds to a turbine,
            would pick the clicked turbine as the target, loading
            up this chart in the first place.
        surrogation_strategies (pandas.DataFrame): The result of
            `load_surrogation_strategies`.
        filtered_power_curves (pandas.DataFrame): The filtered
            dataset stemming from the output of the function
            `load_power_curve_data`.

    Returns:
        fig (plotly.graph_objects.Figure): A Plotly figure.

    """
    colorscale = px.colors.qualitative.Vivid
    fig = go.Figure()
    if selected_neighbors is not None:
        for index, neighbor in enumerate(selected_neighbors):
            r2_value = surrogation_strategies.loc[
                (surrogation_strategies["target"] == selected_target)
                & (surrogation_strategies["surrogate"] == neighbor),
                "bulk_R2",
            ].values[0]

            color = colorscale[index]
            ws_bins = filtered_power_curves.columns.astype(float)
            fig.add_trace(
                go.Scatter(
                    x=ws_bins,
                    y=filtered_power_curves.loc[
                        neighbor, : filtered_power_curves.columns[MAGIC_COLUMN]
                    ].values,
                    mode="lines",
                    marker=dict(color=color),
                    name=f"{neighbor} (R²: {r2_value:.2f})",
                    text=neighbor,
                    hoverinfo="x+y+text",
                    line=dict(dash="solid", shape="spline", width=1),
                )
            )

    # Add the target turbine power curve
    fig.add_trace(
        go.Scatter(
            x=filtered_power_curves.columns.astype(float),  # Wind speed bins
            y=filtered_power_curves.loc[selected_target].values,
            mode="lines",
            name=selected_target,
            marker=dict(color=PARETO_COLORS["highlight"]),
            hoverinfo="x+y",
            line=dict(shape="spline"),
        )
    )
    if selected_neighbors is None:
        title = f"{selected_target}. Neighbors are not available"
    else:
        title = f"{selected_target} compared to {len(selected_neighbors)} Neighbors"
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Average Power (kW)",
        hovermode="x unified",
        xaxis=dict(range=WS_RANGE),
        height=DEFAULT_CHART_HEIGHT,
    )
    return fig
