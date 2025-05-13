"""Module for the Yaw Chart."""

import math

import numpy as np
import plotly.graph_objects as go

from Utils.Transformers import (
    get_turbine,
    get_project_data,
)
from Utils.UiConstants import (
    DEFAULT_CHART_HEIGHT,
)


def aggregate_yaw_data(
    yaw_error_data,
    start=None,
    end=None,
    project=None,
):
    """Aggregate the Yaw data by mean so we can easily plot it.

    Args:
        yaw_error_data (pandas.DataFrame): The index is the date, the columns
            are turbine names, and values are their respective yaw errors.
        start (Optional[datetime]): Start date of the time window to consider.
            Defaults to None.
        end (Optional[datetime]): End date of the time window to consider.
            Defaults to None.
        project (str): The project to filter the data herein by. Set to "All"
            if no filtering is desired.
    """

    # Setting default values for start and end if they're not provided
    if start is None:
        start = yaw_error_data.index[0]
    if end is None:
        end = yaw_error_data.index[-1]

    # Filtering the DataFrame based on start and end, then taking the mean across dates
    chart_data = yaw_error_data.loc[start:end].mean()
    chart_data = chart_data.to_frame().reset_index()
    chart_data.columns = ["Turbine", "Severity"]

    chart_data["Turbine"] = chart_data["Turbine"].apply(lambda x: f"{get_turbine(x)}")

    if project != "All":
        chart_data = get_project_data(chart_data, project)

    # Rename columns to match the data
    chart_data.rename(columns={"Severity": "Angle"}, inplace=True)
    return chart_data


def generate_yaw_chart(yaw_error_data, severity_df, start=None, end=None, project=None):
    """Generates a Yaw Chart.

    Args:
        yaw_error_data (pandas.DataFrame): The index is the date, the columns
            are turbine names, and values are their respective yaw errors.

            For instance:
            ```
                           Turbine1  Turbine2
            Date
            2023-01-01     10        45
            2023-01-02     11        40
            ```
        severity_df (pandas.DataFrame): A dataset of the severity data.
            Note that this value is meant to come directly from the output of the
            `compute_severity_dataset` helper function. See the related docstring
            to learn more about the shape of the data.
        start (Optional[datetime]): Start date of the time window to consider.
            Defaults to None.
        end (Optional[datetime]): End date of the time window to consider.
            Defaults to None.
        project (str): The project to filter the data herein by. Set to "All"
            if no filtering is desired.
    """

    def _gen_xaxis_ticks(min_val, max_val, dtick, extreama_rounding=None):
        """Create an intelligent array for Plotly ticks.

        This function creates a list of numbers uniformly
        spaced by `dtick`. Furthermore, it also explicitly
        includes the min and max values provided.

        Args:
            min (int): Intended to be the minimum value from
                the dataset.
            max (int): Intended to be the minimum value from
                the dataset.
            dtick (int): The normal space between ticks.
            extreama_rounding (int, optional): The number of
                decimal places to round the min and max values
                as they appear in the returned object.
                The default value is 4.
        Returns:
            tickvals (list of int): The tick values. These are
                meant to be passed into the layout.xaxis.tickvals
                property of a Plotly Figure.
        """
        if extreama_rounding is None:
            extreama_rounding = 2

        if np.isnan(min_val):
            min_val = 0
        if np.isnan(max_val):
            max_val = 1

        tickvals = np.arange(math.floor(min_val), math.ceil(max_val), dtick)
        tickvals = tickvals.tolist()
        tickvals.append(math.ceil(max_val))
        tickvals[0] = min_val
        tickvals[-1] = max_val

        rounded_tickvals = []
        for idx, v in enumerate(tickvals):
            decimals = 0
            if idx in (0, len(tickvals) - 1):
                decimals = extreama_rounding
            rounded_tickvals.append(round(v, decimals))
        return rounded_tickvals

    aggr_yaw_data = aggregate_yaw_data(
        yaw_error_data=yaw_error_data,
        start=start,
        end=end,
        project=project,
    )

    # merge the severity data with the yaw data
    merged_df = aggr_yaw_data.merge(severity_df, on="Turbine")
    efficiency_arr = round(merged_df["Severity"] * 100, 1).tolist()
    
    merged_df["Angle"] = merged_df["Angle"].round(
        3
    )  # Rounds the number for the angle to 3 decimal places - FBB 12/23

    # create the figure
    bar_color = "#EEE"
    trace = go.Scatter(
        x=merged_df["Angle"],
        y=[0 for _ in merged_df["Angle"]],
        mode="markers",
        showlegend=False,
        error_y=dict(
            type="data",
            symmetric=False,
            width=0,
            array=efficiency_arr,
            arrayminus=[0 for _ in merged_df["Angle"]],
        ),
    )
    fig = go.Figure(data=[trace], layout=go.Layout(template="plotly_dark"))

    fig.update_traces(
        marker=dict(
            size=5,
            color=bar_color,
            opacity=0.8,
        ),
        line=dict(color=bar_color),
        hoverinfo="text",
        text=merged_df.apply(
            lambda row: f"Turbine: {row['Turbine']}<br>Efficiency: {round(row['Severity'] * 100,1)}%",
            axis=1,
        ),
        hovertemplate="%{text}<extra></extra>",
    )

    # Add a vertical line at x=0° for better chart sense-making
    try:
        max_y = max(efficiency_arr)
    except ValueError:
        max_y = 1

    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=[0, 0],
            y=[0, max_y],
            line=dict(dash="solid", width=3),
            name="Error at 0°",
            hoverinfo="none",
        )
    )

    annotation_color = "#AAA"
    annotations = [
        dict(
            x=angle,
            y=0,
            yshift=-40,
            text=turbine,
            showarrow=False,
            textangle=270,
            font=dict(size=10, color=annotation_color),
        )
        for angle, turbine in zip(merged_df["Angle"], merged_df["Turbine"])
    ]

    min_x = merged_df["Angle"].min()
    max_x = merged_df["Angle"].max()
    tickvals = _gen_xaxis_ticks(
        min_val=min_x,
        max_val=max_x,
        dtick=1,
        extreama_rounding=4,
    )
    fig.update_layout(
        legend=dict(orientation="h"),
        height=DEFAULT_CHART_HEIGHT,
        annotations=annotations,
        hovermode="x",
        xaxis=dict(
            zerolinecolor="rgba(0,0,0,0)",
            title="Yaw Error (degrees)",
            showspikes=True,
            spikethickness=0.25,
            spikecolor="grey",
            spikedash="dot",
            ticksuffix="°",
            tickvals=tickvals,
            rangeslider=dict(
                visible=True,
                bordercolor="#444444",
                bgcolor="#1A1A1A",
                thickness=0.16,
                borderwidth=1,
            ),
        ),
        yaxis=dict(
            visible=True,
            title="Efficiency (%)",
            showgrid=False,
        ),
    )
    return fig
