from datetime import datetime

import dash
from dash import html, ctx, dcc, callback, Input, Output, State, Patch
import pandas as pd

from Charts.Heatmap import (
    generate_fault_heatmap,
)
from Charts.Treemap import (
    generate_fault_treemap,
)
from Charts.Plotters import (
    default_chart,
    generate_pebble_chart,
    generate_pulse_pareto_chart,
)
from Utils.Components import (
    acknowledge_control,
)
from Utils.Transformers import (
    format_date_for_filename,
    join_fault_descriptions,
    add_turbine_fault_column,
)
from Utils.Loaders import (
    load_fault_metrics,
    load_fault_daily_metrics,
    load_fault_code_lookup,
)
from Utils.UiConstants import (
    TURBINE_FAULT_CODE_COUNT,
    PEBBLE_CHART_METRICS,
    PARETO_COLORS,
    ALL_FAULT_CHART_METRICS,
    DESCRIPTION_CODE_DELIM,
)


dash.register_page(
    __name__,
    path="/fault-analysis",
    title="Fault Analysis",
)


def layout():
    output = [
        dcc.Download(id="fault-treemap-download-reciever"),
        dcc.Store(id="start-date-clean", data=None),
        dcc.Store(id="end-date-clean", data=None),
        html.Div(
            className="card",
            children=[
                html.Div(
                    className="fault-control-box",
                    children=dcc.RadioItems(
                        id="pebble-chart-metric",
                        className="pebble-chart-metric",
                        options=[
                            {"label": val, "value": val} for val in PEBBLE_CHART_METRICS
                        ],
                        value=PEBBLE_CHART_METRICS[0],
                        labelStyle={"display": "inline-block"},
                    ),
                ),
                dcc.Graph(
                    id="pebble-chart",
                    figure=default_chart(bgcolor="#171717"),
                ),
            ],
        ),
        html.Div(
            className="card",
            children=[
                html.Div(
                    className="fault-control-box",
                    children=[
                        html.Button(
                            id="download-fault-treemap-btn",
                            className="download-btn",
                            children="Download",
                        ),
                        dcc.RadioItems(
                            id="fault-treemap-metric",
                            options=[
                                {"label": val, "value": val}
                                for val in ALL_FAULT_CHART_METRICS
                            ],
                            value=ALL_FAULT_CHART_METRICS[0],
                            labelStyle={"display": "inline-block"},
                        ),
                    ],
                ),
                html.Div(
                    id="treemap-heatmap-box2",
                    className="treemap-subcharts-container",
                    children=[
                        dcc.Graph(
                            id="fault-treemap",
                            figure=default_chart(bgcolor="#171717"),
                        ),
                        html.Div(
                            id="heatmap-box2",
                            className="is-closed",
                            children=[
                                dcc.RadioItems(
                                    id="heatmap-toggle2",
                                    className="heatmap-toggle",
                                    options=[
                                        {
                                            "label": "By Fault",
                                            "value": "by-fault",
                                        },
                                        {
                                            "label": "By Turbine",
                                            "value": "by-turbine",
                                        },
                                    ],
                                    value="by-turbine",
                                    labelStyle={"display": "inline-block"},
                                ),
                                dcc.Graph(
                                    id="temp-heatmap2",
                                    figure={"data": [{"x": [1, 2, 3], "y": [4, 5, 6]}]},
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="card",
            children=[
                html.Div(
                    className="title-and-controls",
                    children=[
                        html.Span(
                            className="chart-title",
                            children=[
                                html.Span(
                                    children="Fault Timeline",
                                    className="title",
                                ),
                                html.Span(
                                    children="",
                                    className="subtitle",
                                ),
                            ],
                        ),
                        html.Div(
                            className="chart-control",
                            children=[
                                dcc.RadioItems(
                                    id="pulse-pareto-metric",
                                    className="right-floating-control",
                                    options=[
                                        {"label": val, "value": val}
                                        for val in ALL_FAULT_CHART_METRICS
                                    ],
                                    value=ALL_FAULT_CHART_METRICS[0],
                                    labelStyle={"display": "inline-block"},
                                ),
                                acknowledge_control(
                                    _id="acknowledged-turbine-fault-pairs",
                                    label="Turbine-Faults to Hide",
                                    placeholder="Hide Turbine-Faults Pairs from the chart...",
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Graph(
                    id="pulse-pareto-chart",
                    figure=default_chart(bgcolor="#171717"),
                ),
            ],
        ),
    ]
    return output


@callback(
    Output("pebble-chart", "figure"),
    Input("date-rangeslider", "value"),
    Input("pebble-chart-metric", "value"),
    Input("project-dropdown", "value"),
    Input("oem-dropdown", "value"),
    Input("acknowledged-fault-descriptions", "value"),
    State("date-intervals-store", "data"),
)
def update_pebble_chart(
    date_value_idx,
    metric,
    project,
    oem,
    acknowledged_faults,
    date_intervals_store,
):
    df_metric = load_fault_daily_metrics()
    fault_description_df = load_fault_code_lookup()

    start = date_value_idx[0]
    end = date_value_idx[1]
    if start is not None:
        start_date = date_intervals_store[start]
        start_date = start_date.split("T")[0]
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if end is not None:
        end_date = date_intervals_store[end]
        end_date = end_date.split("T")[0]
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    pebble_chart = generate_pebble_chart(
        df_metric=df_metric,
        fault_description_df=fault_description_df,
        metric=metric,
        start=start_date,
        end=end_date,
        project=project,
        oem=oem,
        acknowledged_faults=acknowledged_faults,
    )
    return pebble_chart


@callback(
    Output("fault-treemap", "figure"),
    Output("start-date-clean", "data"),
    Output("end-date-clean", "data"),
    Input("date-rangeslider", "value"),
    Input("fault-treemap-metric", "value"),
    Input("project-dropdown", "value"),
    Input("oem-dropdown", "value"),
    Input("acknowledged-fault-descriptions", "value"),
    State("date-intervals-store", "data"),
)
def update_fault_treemap(
    date_value_idx, metric, project, oem, acknowledged_faults, date_intervals_store
):
    df_metric = load_fault_daily_metrics()
    fault_description_df = load_fault_code_lookup()

    start = date_value_idx[0]
    end = date_value_idx[1]
    if start is not None:
        start_date = date_intervals_store[start]
        start_date = start_date.split("T")[0]
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if end is not None:
        end_date = date_intervals_store[end]
        end_date = end_date.split("T")[0]
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    fault_treemap = generate_fault_treemap(
        df_metric=df_metric,
        fault_description_df=fault_description_df,
        metric=metric,
        start=start_date,
        end=end_date,
        project=project,
        oem=oem,
        acknowledged_faults=acknowledged_faults,
    )
    return fault_treemap, start_date, end_date


@callback(
    Output("temp-heatmap2", "figure"),
    Output("heatmap-box2", "className"),
    Output("treemap-heatmap-box2", "className"),
    Input("date-rangeslider", "value"),
    Input("fault-treemap", "clickData"),
    Input("heatmap-toggle2", "value"),
    Input("fault-treemap-metric", "value"),
    Input("project-dropdown", "value"),
    State("heatmap-box2", "className"),
    State("date-intervals-store", "data"),
    prevent_inital_call=True,
)
def toggle_fault_treemap_subcharts(
    date_value_idx,
    treemapClickData,
    heatmap_toggle,
    metric,
    project_dropdown,
    last_heatmap_cls,
    date_intervals_store,
):
    heatmap_cls = last_heatmap_cls
    if ctx.triggered_id is None:
        return dash.no_update
    elif ctx.triggered_id == "fault-treemap":
        if last_heatmap_cls == "is-closed":
            heatmap_cls = "is-open"
        else:
            heatmap_cls = "is-closed"
    elif ctx.triggered_id in (
        "date-rangeslider",
        "fault-treemap-metric",
        "project-dropdown",
    ):
        heatmap_cls = "is-closed"

    if heatmap_cls == "is-closed":
        heatmap = default_chart(bgcolor="#171717")
        treemap_cls = "treemap-subcharts-container"
        return heatmap, heatmap_cls, treemap_cls

    label = treemapClickData["points"][0]["label"]
    project_turbine, fault_description = label.split("<br>")
    project = project_turbine.split("-")[0]
    turbine = project_turbine.split("-")[1]

    fault_description_df = load_fault_code_lookup()
    daily_dataset = load_fault_daily_metrics()

    start = date_value_idx[0]
    end = date_value_idx[1]
    if start is not None:
        start_date = date_intervals_store[start]
        start_date = pd.to_datetime(start_date)
    if end is not None:
        end_date = date_intervals_store[end]
        end_date = pd.to_datetime(end_date)
    if heatmap_toggle == "by-fault":
        dataset = load_fault_metrics()
        chart = generate_pulse_pareto_chart(
            fault_metrics_df=dataset,
            fault_daily_metrics_df=daily_dataset,
            fault_description_df=fault_description_df,
            start_date=start_date,
            end_date=end_date,
            metric=metric,
            project=project,
            turbine_arr=[project_turbine],
        )
        chart.update_layout(
            height=375,
        )
    elif heatmap_toggle == "by-turbine":
        dataset = load_fault_daily_metrics()
        chart = generate_fault_heatmap(
            dataset=dataset,
            fault_description_df=fault_description_df,
            metric=metric,
            fault_description=fault_description,
        )
    treemap_cls = "treemap-subcharts-container bottom-margin2"
    return chart, heatmap_cls, treemap_cls


@callback(
    Output("pulse-pareto-chart", "figure"),
    Input("date-rangeslider", "value"),
    Input("pulse-pareto-metric", "value"),
    Input("project-dropdown", "value"),
    Input("acknowledged-turbine-fault-pairs", "value"),
    Input("acknowledged-fault-descriptions", "value"),
    Input("oem-dropdown", "value"),
    State("date-intervals-store", "data"),
    State("pulse-pareto-chart", "figure"),
)
def update_pulse_pareto_chart(
    date_value_idx,
    metric,
    project,
    ack_turbine_fault_pairs,
    ack_fault_descr_pairs,
    oem,
    date_intervals_store,
    lastFigure,
):
    def _highlight_pulse_row(hoverData, lastFigure, patched_figure):
        """Highlights Pulse row inline with Pareto bar you hover over."""
        turbine_fault_on_hover = hoverData["points"][0]["y"]

        for idx, trace in enumerate(lastFigure["data"]):
            color = PARETO_COLORS["normal"]
            if trace["name"] in (turbine_fault_on_hover, "pareto-trace"):
                color = PARETO_COLORS["highlight"]
            patched_figure.data[idx]["marker"]["color"] = color
        return patched_figure

    if dash.callback_context.triggered_id == "pulse-pareto-chart":
        patched_figure = Patch()
        patched_figure = _highlight_pulse_row(
            hoverData=hoverData,
            lastFigure=lastFigure,
            patched_figure=patched_figure,
        )
        return patched_figure

    dataset = load_fault_metrics()
    daily_dataset = load_fault_daily_metrics()
    fault_description_df = load_fault_code_lookup()
    
    start = date_value_idx[0]
    end = date_value_idx[1]
    if start is not None:
        start_date = date_intervals_store[start]
        start_date = pd.to_datetime(start_date)
    if end is not None:
        end_date = date_intervals_store[end]
        end_date = pd.to_datetime(end_date)

    chart = generate_pulse_pareto_chart(
        fault_metrics_df=dataset,
        fault_daily_metrics_df=daily_dataset,
        fault_description_df=fault_description_df,
        start_date=start_date,
        end_date=end_date,
        metric=metric,
        project=project,
        num_of_rows=TURBINE_FAULT_CODE_COUNT,
        ack_turbine_fault_pairs=ack_turbine_fault_pairs,
        ack_fault_descr_pairs=ack_fault_descr_pairs,
        oem=oem,
    )
    return chart


@callback(
    Output("acknowledged-turbine-fault-pairs", "options"),
    Input("url", "pathname"),
)
def populate_pareto_acknowledge_dropdown_options(pathname):
    """The options are of the form '{turbine} - {code}', ex. 'BR2-K001 - 2'."""
    parsed_pathname = dash.strip_relative_path(pathname)
    if parsed_pathname == "fault-analysis":
        fault_metrics_df = load_fault_metrics()
        fault_metrics_df = add_turbine_fault_column(fault_metrics_df)

        turbine_fault_codes = fault_metrics_df["TurbineFaultCode"].unique()
        return turbine_fault_codes
    return dash.no_update


@callback(
    Output("acknowledged-fault-descriptions", "options"),
    Input("url", "pathname"),
)
def populate_global_fault_acknowledge_dropdown_options(pathname):
    """The options are of the form '{code} | {description}', ex. '5 | REPAIR'."""
    parsed_pathname = dash.strip_relative_path(pathname)
    if parsed_pathname == "fault-analysis":
        fault_metrics_df = load_fault_metrics()
        fault_metrics_df = add_turbine_fault_column(fault_metrics_df)

        fault_description_df = load_fault_code_lookup()
        combined_df = join_fault_descriptions(
            data_frame=fault_metrics_df,
            fault_description_df=fault_description_df,
            code_colname="FaultCode",
        )

        combined_df["DescriptionFaultCode"] = (
            combined_df["FaultCode"].astype(int).astype(str)
            + DESCRIPTION_CODE_DELIM
            + combined_df["Description"]
        )
        combined_df.sort_values(by="FaultCode", ascending=True, inplace=True)
        return combined_df["DescriptionFaultCode"].unique()
    return dash.no_update


@callback(
    Output("fault-treemap-download-reciever", "data"),
    Input("download-fault-treemap-btn", "n_clicks"),
    State("fault-treemap", "figure"),
    State("project-dropdown", "value"),
    State("start-date-clean", "data"),
    State("end-date-clean", "data"),
    prevent_initial_call=True,
)
def download_fault_treemap_dataset(
    n_clicks, fault_treemap_json, project, start_date, end_date
):
    if n_clicks is None:
        return dash.no_update

    num_of_rows = 10
    customdata_order = (
        "Turbine",
        "Fault",
        "Downtime (Hours)",
        "Lost Revenue ($)",
        "Lost Energy (MWh)",
        "Count",
    )

    customdata = fault_treemap_json["data"][0]["customdata"]
    df_rows = []
    for row in customdata[:num_of_rows]:
        df_rows.append({customdata_order[idx]: row[idx] for idx in range(len(row))})
    df = pd.DataFrame(df_rows)

    start_date_fmt = format_date_for_filename(start_date)
    end_date_fmt = format_date_for_filename(end_date)

    project_fmt = project
    if project == "All":
        project_fmt = "all_projects"
    project_fmt = project_fmt.replace(" ", "")
    filename = (
        f"{start_date_fmt}_to_{end_date_fmt}_treemap_fault_dataset_{project_fmt}.xlsx"
    )

    return dcc.send_data_frame(df.to_excel, filename, sheet_name="Sheet1")
