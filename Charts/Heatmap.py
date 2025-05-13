"""A module that contains Heatmap Charts used in the App.

Heatmaps is a data visualization, essentially a 2D array of cells which
each are associated with data. In the application, heatmaps exclusively
appear inside and after clicking on a Treemap's cell. They are used to
display a more drilled-down version of the clicked cell.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Charts.Hovertemplate import (
    append_units_and_round,
)
from Utils.Constants import (
    DEFAULT_PARSE_FUNCS,
)
from Utils.Transformers import (
    join_fault_descriptions,
    get_turbine,
)
from Utils.UiConstants import (
    DEFAULT_CHART_HEIGHT,
    REV_FAULT_METRIC_COLUMN_LOOKUP,
)


def make_styled_heatmap(z, x, y, hovertemplate=None, customdata=None, height=None):
    """A wrapper around Plotly's Heatmap that adds styling and coloring.

    Args:
        z: See plotly.graph_objects.Heatmap.z.
        x: See plotly.graph_objects.Heatmap.x.
        y: See plotly.graph_objects.Heatmap.y.
        hovertemplate: See plotly.graph_objects.Heatmap.hovertemplate.
        customdata: See plotly.graph_objects.Heatmap.hovertemplate.
        height: See plotly.graph_objects.Figure.layout.height.
    Returns:
        fig (plotly.graph_objects.Figure): The styled heatmap.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="YlOrRd_r",
            hovertemplate=hovertemplate,
            customdata=customdata,
            colorbar=dict(
                orientation="h",
                thickness=10,
            ),
        )
    )
    fig.update_layout(
        height=height,
        template="plotly_dark",
        margin=dict(
            autoexpand=True,
            t=45,
            l=10,
            r=50,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=12), tickangle=270)
    fig.update_yaxes(side="right", gridcolor="rgba(0,0,0,0)")
    return fig


def generate_comp_temp_heatmap(data_frame, lost_energy_df=None, mean_frame=None):
    """Make heatmap that appears upon drilling down on component treemap cell.

    Args:
        data_frame (pandas.DataFrame): The index is the date, the columns
            are turbine names, and values are their respective Severities.
        lost_energy_df (pandas.DataFrame): This comes from the
            output of the `filter_lost_energy` function. See its docstring
            for relevant info about its properties.
    Returns:
        (plotly.graph_objects.Figure): A Plotly heatmap.
    """
    fig = go.Figure()
    data_frame = data_frame.dropna(axis=1, how="all").drop_duplicates()
    data_frame = data_frame.sort_index(axis=1)

    # ensure both dataframes have same column order
    parse_turbine = get_turbine
    parse_comp = DEFAULT_PARSE_FUNCS["component_type_func"]

    # lost_energy is a proxy for by turbine
    if lost_energy_df is not None:
        x_arr = [parse_turbine(name) for name in data_frame]
        data_frame.sort_index(axis=1, inplace=True)

        row_mean = mean_frame.mean(axis=1)

        
        result_df = pd.DataFrame([row_mean.drop_duplicates()] * len(mean_frame.columns)).T



        result_df = result_df.sort_index(axis=1)
        park_avg = result_df.values

        for col in mean_frame.columns:
            col = col.replace("_mean", "")
            col = get_turbine(col) + "-Lost_Energy"
            if col not in lost_energy_df.columns:
                lost_energy_df[col] = [0] * len(mean_frame)

        lost_energy_df.sort_index(axis=1, inplace=True)

        component = parse_comp(data_frame.columns[0])

        # Create customdata
        customdata = [
            [mean, park, energy]
            for park, energy, mean in zip(
                park_avg, lost_energy_df.values, mean_frame.values
            )
        ]
        customdata_3D = np.array(customdata)
        customdata = np.transpose(customdata_3D, (0, 2, 1))

        hovertemplate = (
            "Turbine: %{x}<br>"
            f"Component: {component}<br>"
            "Date: %{y}<br>"
            "Mean Temperature: %{customdata[0]:.0f}°C<br>"
            "Park Average Temperature: %{customdata[1]:.0f}°C<br>"
            "Lost Energy: %{customdata[2]:.0d}MWh"
        )

    else:
        x_arr = [parse_comp(name) for name in data_frame]
        turbine = parse_turbine(data_frame.columns[0])

        transposed_df = mean_frame.T.dropna(how="all")
        mean_frame = transposed_df.T
        transposed_df["component_type"] = transposed_df.index.str.split("-").str[-1]
        park_avg_by_component = transposed_df.groupby("component_type").mean()

        # Broadcast the mean values to match the shape of the original dataframe
        result_df = mean_frame.copy()
        for col in mean_frame.columns:
            component = col.split("-")[-1]
            if component not in park_avg_by_component.index:
                result_df[col] = 0
            else:
                result_df[col] = park_avg_by_component.loc[component, :]

        result_df = result_df.sort_index(axis=True)

        # filter down to only the turbine that we need for performance
        result_df = result_df.loc[:, result_df.columns.str.contains(turbine)]
        mean_frame = mean_frame.loc[:, mean_frame.columns.str.contains(turbine)]
        data_frame = data_frame.loc[:, data_frame.columns.str.contains(turbine)]

        park_avg = result_df.values

        # Create customdata
        customdata = [[mean, park] for park, mean in zip(park_avg, mean_frame.values)]
        customdata_3D = np.array(customdata)
        customdata = np.transpose(customdata_3D, (0, 2, 1))

        hovertemplate = (
            f"Turbine: {turbine}<br>"
            "Component: %{x}<br>"
            "Date: %{y}<br>"
            "Mean Temperature: %{customdata[0]:.0f}°C<br>"
            "Park Average Temperature: %{customdata[1]:.0f}°C"
        )
    
    # print(data_frame.columns[0])
    # print('z vals',np.shape(data_frame.values))
    # print('x vals', np.shape(x_arr))
    # print('y vals', np.shape(data_frame.index))
    # data_frame.to_csv(data_frame.columns[0])
    hovertemplate += "<extra></extra>"
    fig = make_styled_heatmap(
        z=data_frame.values,
        x=x_arr,
        y=data_frame.index,
        customdata=customdata,
        hovertemplate=hovertemplate,
        height=DEFAULT_CHART_HEIGHT + 100,
    )
    return fig


def generate_fault_heatmap(
    dataset,
    fault_description_df,
    metric,
    fault_description,
):
    """Displays Turbines across time sharing the same fault code.

    Args:
        dataset (pandas.DataFrame): The dataset that contains the fault metrics.
            This dataset should come from the output of the `load_fault_daily_metrics`.
        fault_description_df (pd.DataFrame): This is the dataset that contains the
            Fault Code descriptions.
        metric (str): One of "Downtime", "Lost Energy", "Lost Revenue", or "Count".
        fault_description (str): The fault code description. eg. "REPAIR"
        start_date (datetime): The start date of the chart.
        end_date (datetime): The end date of the chart.

    Returns:
        fig (plotly.graph_objects.Figure): A Plotly heatmap.
    """
    fmt_metric = REV_FAULT_METRIC_COLUMN_LOOKUP[metric]

    dataset = join_fault_descriptions(
        data_frame=dataset,
        fault_description_df=fault_description_df,
        code_colname="FaultCode",
    )
    dataset = dataset[dataset["Description"] == fault_description]
    dff = dataset[["Date", "Turbine", fmt_metric]]
    pivot_df = dff.pivot(index="Date", columns="Turbine", values=fmt_metric)

    z_hover = append_units_and_round(var="z", metric=metric)
    hovertemplate = f"Turbine: %{{x}}<br>Date: %{{y}}<br>{z_hover}<extra></extra>"
    fig = make_styled_heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        hovertemplate=hovertemplate,
    )
    return fig
