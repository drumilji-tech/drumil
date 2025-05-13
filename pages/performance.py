from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import ctx, dcc, html, callback, Input, Output, State

from Charts.Heatmap import generate_comp_temp_heatmap
from Charts.Plotters import default_chart
from Charts.PowerCurve import (
    gen_level1_subcharts,
    gen_peer_to_peer_chart,
)
from Charts.Treemap import (
    aggregate_pp_treemap_data,
    generate_comp_temp_treemap,
    generate_power_performance_treemap,
)
from Charts.PowerCurve import (
    calculate_weighted_average_power_by_turbine,
    calculate_weighted_average_power_collapse_bins,
    calculate_AEP,
    full_calculate_AEP,
    find_common_valid_indices,
)
from Charts.Yaw import generate_yaw_chart
from Utils.Constants import DEFAULT_PARSE_FUNCS
from Utils.Components import (
    gen_date_intervals,
    acknowledge_control,
    gen_table_component,
)
from Utils.Loaders import (
    load_simple_efficiency_dataset,
    load_treemap_dataset,
    load_yaw_error_dataset,
    load_power_curve_data,
    load_power_distribution_data,
    load_ws_distribution_data,
    load_surrogation_strategies,
)
from Utils.Transformers import (
    format_date_for_filename,
    filter_treemap_columns,
    filter_lost_energy,
    filter_mean_values,
    compute_severity_dataset,
    component_type_map,
    format_columns,
)
from Utils.UiConstants import (
    PERFORMANCE_METRICS,
    POWER_PERFORMANCE_TREEMAP_OPTIONS,
)


TOOLTIP_DELAY_TIMINGS = {
    "show": 600,
    "hide": 750,
}
TOOLTIP_TEXT_LOOKUP = {
    PERFORMANCE_METRICS[0]: (
        "The amount of money lost or gained relative to what was expected "
        "from the turbine over the selected period of time. The cost per "
        "MWh varies every month and for every site."
    ),
    PERFORMANCE_METRICS[1]: (
        "The amount of energy, in MWh, that is not generated relative to "
        "what was expected from the turbine over the selected period of "
        "time. No other variables influence this metric.",
    ),
    PERFORMANCE_METRICS[2]: (
        "A measurement of the deviation in performance relative to the rest "
        "of the wind farm (and class) where the turbine belongs over the "
        "selected period of time. Higher severity indicates higher "
        "separation between actual and expected power than the other turbines "
        "in the site.",
    ),
}


treemap_data_simple_eff = load_simple_efficiency_dataset()
treemap_data_from_file = load_treemap_dataset()
yaw_error_data = load_yaw_error_dataset()
ws_dist = load_ws_distribution_data()

dash.register_page(
    __name__,
    path="/performance-and-reliability",
    title="Performance and Reliability",
)


def layout():
    component_temperature_graph = dcc.Graph(
        id="treemap", className="treemap", figure=default_chart()
    )
    power_performance_graph = dcc.Graph(id="pp-treemap", figure=default_chart())
    temp_heatmap_chart = dcc.Graph(
        id="temp-heatmap",
        figure=default_chart(bgcolor="#171717"),
        config={"displayModeBar": False},
    )

    heatmap_box = html.Div(
        id="heatmap-box",
        className="is-closed",
        children=[
            dcc.RadioItems(
                id="heatmap-toggle",
                className="heatmap-toggle",
                options=[
                    {
                        "label": "By Turbine",
                        "value": "by-turbine",
                    },
                    {
                        "label": "By Component",
                        "value": "by-component",
                    },
                ],
                value="by-turbine",
                labelStyle={"display": "inline-block"},
            ),
            temp_heatmap_chart,
        ],
    )

    pp_treemap_and_subcharts = html.Div(
        id="treemap-heatmap-box3",
        className="treemap-subcharts-container",
        children=[
            power_performance_graph,
            html.Div(
                id="heatmap-box3",
                className="is-closed",
                children=[
                    dcc.Tabs(
                        id="heatmap-toggle3",
                        className="heatmap-toggle",
                        value="level-1",
                        children=[
                            dcc.Tab(
                                label="Level 1",
                                value="level-1",
                                children=[
                                    dcc.Graph(
                                        id="level-1-subplot",
                                        figure=default_chart("#171717"),
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="Level 2",
                                value="level-2",
                                children=[
                                    html.Span("Neighboring Turbines Shown"),
                                    dcc.Dropdown(
                                        id="neighbors-dropdown",
                                        multi=True,
                                    ),
                                    dcc.Graph(
                                        id="peer2peer-chart",
                                        figure=default_chart("#171717"),
                                    ),
                                    gen_table_component(
                                        _id="peer-to-peer-table",
                                        table_columns=("Pairings", "% AEP Delta"),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    output = [
        dcc.Download(id="temp-treemap-download-receiver"),
        dcc.Store(id="power-perf-click-store", data=None),
        dcc.Store(id="start-date-clean2", data=None),
        dcc.Store(id="end-date-clean2", data=None),
        html.Div(
            className="card",
            children=[
                html.Div(
                    className="title-and-controls",
                    children=[
                        html.Span(
                            id="pp-treemap-title",
                            className="chart-title",
                        ),
                        html.Div(
                            className="chart-control",
                            children=[
                                dbc.Tooltip(
                                    TOOLTIP_TEXT_LOOKUP[PERFORMANCE_METRICS[0]],
                                    target=PERFORMANCE_METRICS[0],
                                    placement="bottom",
                                    delay=TOOLTIP_DELAY_TIMINGS,
                                    class_name="metric-tooltip",
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_TEXT_LOOKUP[PERFORMANCE_METRICS[1]],
                                    target=PERFORMANCE_METRICS[1],
                                    placement="bottom",
                                    delay=TOOLTIP_DELAY_TIMINGS,
                                    class_name="metric-tooltip",
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_TEXT_LOOKUP[PERFORMANCE_METRICS[2]],
                                    target=PERFORMANCE_METRICS[2],
                                    placement="bottom",
                                    delay=TOOLTIP_DELAY_TIMINGS,
                                    class_name="metric-tooltip",
                                ),
                                dcc.RadioItems(
                                    id="sort-by",
                                    className="right-floating-control",
                                    options=POWER_PERFORMANCE_TREEMAP_OPTIONS,
                                    value=POWER_PERFORMANCE_TREEMAP_OPTIONS[0]["value"],
                                    labelStyle={"display": "inline-block"},
                                ),
                                dcc.RadioItems(
                                    id="under-over-perform",
                                    className="under-over-perform",
                                    options=[
                                        {
                                            "label": "Overperforming",
                                            "value": "overperforming",
                                        },
                                        {
                                            "label": "Underperforming",
                                            "value": "underperforming",
                                        },
                                    ],
                                    value="underperforming",
                                    labelStyle={"display": "inline-block"},
                                ),
                                acknowledge_control(_id="acknowledged-pp-turbines"),
                            ],
                        ),
                    ],
                ),
                pp_treemap_and_subcharts,
            ],
        ),
        html.Div(
            className="card",
            children=[
                html.Div(
                    className="title-and-controls",
                    children=[
                        html.Span(
                            id="treemap-title",
                            className="chart-title",
                        ),
                        html.Div(
                            className="chart-control",
                            children=[
                                html.Button(
                                    id="download-temp-treemap-btn",
                                    className="download-btn",
                                    children="Download",
                                ),
                                html.Label(
                                    className="dropdown-label",
                                    htmlFor="component-dropdown",
                                    children="Filter By Component",
                                ),
                                dcc.Dropdown(
                                    id="component-dropdown",
                                    className="dropdown-comp",
                                    value="All",
                                    clearable=False,
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="treemap-heatmap-box",
                    className="treemap-subcharts-container",
                    children=[
                        component_temperature_graph,
                        heatmap_box,
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
                            id="yaw-error-chart-title",
                            className="chart-title",
                        ),
                    ],
                ),
                dcc.Graph(
                    id="yaw-error-chart",
                    className="yaw-error-chart",
                    figure=default_chart(),
                ),
            ],
        ),
    ]
    return output


@callback(
    Output("start-date-clean2", "data"),
    Output("end-date-clean2", "data"),
    Input("date-rangeslider", "value"),
    State("date-intervals-store", "data"),
)
def store_dates(date_value_idx, date_intervals_store):
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
    return start_date, end_date


@callback(
    Output("temp-treemap-download-receiver", "data"),
    Input("download-temp-treemap-btn", "n_clicks"),
    State("treemap", "figure"),
    State("project-dropdown", "value"),
    State("component-dropdown", "value"),
    State("start-date-clean2", "data"),
    State("end-date-clean2", "data"),
    prevent_initial_call=True,
)
def download_temperature_treemap_dataset(
    n_clicks,
    treemap_json,
    project,
    component,
    start_date,
    end_date,
):
    if n_clicks is None:
        return dash.no_update

    num_of_rows = 10
    customdata = treemap_json["data"][0]["customdata"]
    labels = treemap_json["data"][0]["labels"]
    project_arr = [l.split("<br>")[0].split("-")[0] for l in labels]
    turbine_arr = [l.split("<br>")[0].split("-")[1] for l in labels]
    component_arr = [l.split("<br>")[1] for l in labels]

    df_rows = []
    for k, temp_container in enumerate(customdata[:num_of_rows]):
        row = {}
        row["Project"] = project_arr[k]
        row["Turbine"] = turbine_arr[k]
        row["Component"] = component_arr[k]
        row["Temperature (°C)"] = temp_container[0]
        row["Park Average (°C)"] = temp_container[1]
        df_rows.append(row)
    df = pd.DataFrame(df_rows)

    start_date_fmt = format_date_for_filename(start_date)
    end_date_fmt = format_date_for_filename(end_date)

    project_fmt = project
    if project == "All":
        project_fmt = "all_projects"
    project_fmt = project_fmt.replace(" ", "")

    component_fmt = "_" + component
    if component == "All":
        component_fmt = ""
    filename = f"{start_date_fmt}_to_{end_date_fmt}_temperature_{project_fmt}{component_fmt}.xlsx"
    return dcc.send_data_frame(df.to_excel, filename, sheet_name="Sheet1")


@callback(
    Output("heatmap-box3", "className"),
    Output("treemap-heatmap-box3", "className"),
    Input("pp-treemap", "clickData"),
    Input("heatmap-toggle3", "value"),
    Input("start-date-clean2", "data"),
    Input("end-date-clean2", "data"),
    Input("date-rangeslider", "value"),
    Input("project-dropdown", "value"),
    Input("under-over-perform", "value"),
    Input("sort-by", "value"),
    Input("acknowledged-pp-turbines", "value"),
    State("heatmap-box3", "className"),
    prevent_inital_call=True,
)
def toggle_power_performance_treemap_subcharts(
    treemapClickData,
    toggle,
    start_date,
    end_date,
    date_range_slider,
    project_dropdown,
    under_over_perform,
    sort_by,
    acknowledged_pp_turbines,
    last_subchart_cls,
):
    subchart_cls = last_subchart_cls
    if ctx.triggered_id is None:
        return dash.no_update
    elif ctx.triggered_id == "pp-treemap":
        if last_subchart_cls == "is-closed":
            subchart_cls = "is-open"
        else:
            subchart_cls = "is-closed"
    elif ctx.triggered_id in (
        "date-rangeslider",
        "project-dropdown",
        "under-over-perform",
        "sort-by",
        "acknowledged-pp-turbines",
    ):
        subchart_cls = "is-closed"

    if subchart_cls == "is-closed":
        treemap_cls = "treemap-subcharts-container"
        return subchart_cls, treemap_cls
    label = treemapClickData["points"][0]["label"]
    project_turbine, component_type = label.split("<br>")

    if toggle == "level-1":
        power_curve_df = load_power_curve_data()
        distribution_df = load_power_distribution_data()
        ws_distribution = load_ws_distribution_data()

        chart = gen_level1_subcharts(
            power_curve_df=power_curve_df,
            distribution_df=distribution_df,
            ws_distribution=ws_distribution,
            turbine=project_turbine,
            start_date=start_date,
            end_date=end_date,
        )
    elif toggle == "level-2":
        chart = {
            "data": [{"x": [1, 2, 3], "y": [1, 2, 3]}],
            "layout": {"title": "Peer to Peer & Month2Month"},
        }

    treemap_cls = "treemap-subcharts-container bottom-margin3"
    return subchart_cls, treemap_cls


@callback(
    Output("component-dropdown", "options"),
    Input("component-dropdown", "value"),
)
def populate_temp_treemap_dropdown_options(value):
    component_list = ["All"] + sorted(
        list(
            set(
                [
                    component_type_map(DEFAULT_PARSE_FUNCS["component_type_func"](x))
                    if component_type_map(DEFAULT_PARSE_FUNCS["component_type_func"](x))
                    is not None
                    else ""
                    for x in treemap_data_from_file.columns
                ]
            )
        )
    )
    dropdown_options = [
        {"label": component, "value": component} for component in component_list
    ]
    return dropdown_options


@callback(
    Output("treemap", "figure"),
    Output("pp-treemap", "figure"),
    Output("treemap-title", "children"),
    Output("pp-treemap-title", "children"),
    Output("yaw-error-chart", "figure"),
    Output("yaw-error-chart-title", "children"),
    Input("date-rangeslider", "value"),
    Input("component-dropdown", "value"),
    Input("project-dropdown", "value"),
    Input("under-over-perform", "value"),
    Input("acknowledged-pp-turbines", "value"),
    Input("sort-by", "value"),
)
def update_charts(
    date_value_idx,
    component_type,
    project,
    under_over_perform,
    acknowledged_pp_turbines,
    sort_by,
):
    date_intervals = gen_date_intervals(treemap_data_from_file)
    start = date_intervals[date_value_idx[0]]
    end = date_intervals[date_value_idx[1]]
    
    # Update treemap
    treemap_fig = generate_comp_temp_treemap(
        treemap_data_from_file=treemap_data_from_file,
        start=start,
        end=end,
        component_type=component_type,
        project=project,
    )

    # Update power chart
    power_fig = generate_power_performance_treemap(
        treemap_data_simple_eff=treemap_data_simple_eff,
        start=start,
        end=end,
        project=project,
        overperforming=under_over_perform,
        acknowledged_pp_turbines=acknowledged_pp_turbines,
        sort_by=sort_by,
    )

    severity_df = compute_severity_dataset(
        treemap_data_from_file=treemap_data_simple_eff.loc[
            :, treemap_data_simple_eff.columns.str.contains("-EFFICIENCY")
        ],
        start=start,
        end=end,
        agg="mean",
    )

    
    yaw_error_fig = generate_yaw_chart(
        yaw_error_data=yaw_error_data,
        severity_df=severity_df,
        start=start,
        end=end,
        project=project,
    )

    # Update title for treemap
    if start is None:
        start = treemap_data_from_file.index[0]
    if end is None:
        end = treemap_data_from_file.index[-1]
    start_str = start.strftime("%B %d, %Y")
    end_str = end.strftime("%B %d, %Y")
    treemap_fig_title = [
        html.Span(
            children=f"Component Temperature ",
            className="title",
        ),
        html.Span(
            children=f"({start_str} to {end_str})",
            className="subtitle",
        ),
    ]

    # Update title for power chart
    if start is None:
        start = treemap_data_simple_eff.index[0]
    if end is None:
        end = treemap_data_simple_eff.index[-1]
    start_str = start.strftime("%B %d %Y")
    end_str = end.strftime("%B %d %Y")
    power_fig_title = [
        html.Span(
            children=f"Power Performance ",
            className="title",
        ),
        html.Span(
            children=f"({start_str} to {end_str})",
            className="subtitle",
        ),
    ]

    # Update the title for Yaw Chart
    project_label = project if project != "All" else "All Projects"
    comp_label = component_type if component_type != "All" else "All Components"
    subtitle = f"- {project_label}, {comp_label}"
    yaw_error_title = [
        html.Span(
            children=f"Yaw Error by Turbine ",
            className="title",
        ),
        html.Span(
            children=subtitle,
            className="subtitle",
        ),
    ]

    return (
        treemap_fig,
        power_fig,
        treemap_fig_title,
        power_fig_title,
        yaw_error_fig,
        yaw_error_title,
    )


@callback(
    Output("acknowledged-pp-turbines", "options"),
    Input("component-dropdown", "value"),
    Input("project-dropdown", "value"),
    Input("under-over-perform", "value"),
    Input("sort-by", "value"),
)
def populate_acknowledge_pp_turbine_options(
    component_type,
    project,
    under_over_perform,
    sort_by,
):
    options_arr = []
    # TODO: Make the valid sort_by options into a constnat and reuse
    for val in PERFORMANCE_METRICS:
        for cutoff in ["underperforming", "overperforming"]:
            treemap_data = aggregate_pp_treemap_data(
                treemap_data_simple_eff=treemap_data_simple_eff,
                start=None,
                end=None,
                project="All",
                overperforming=cutoff,
                sort_by=val,
            )
            names = treemap_data["Turbine"].tolist()
            names = ["-".join(n.split("-")[:2]) for n in names]
            options = [{"label": turbine, "value": turbine} for turbine in names]
            options_arr.extend(options)
    return options_arr


@callback(
    Output("temp-heatmap", "figure"),
    Output("heatmap-box", "className"),
    Output("treemap-heatmap-box", "className"),
    Input("treemap", "clickData"),
    Input("heatmap-toggle", "value"),
    Input("date-rangeslider", "value"),
    Input("project-dropdown", "value"),
    Input("component-dropdown", "value"),
    State("heatmap-box", "className"),
    prevent_inital_call=True,
)
def update_temperature_heatmap(
    treemapClickData,
    heatmap_toggle,
    range_slider,
    project_dropdown,
    component_dropdown,
    last_heatmap_cls,
):
    heatmap_cls = last_heatmap_cls
    if ctx.triggered_id is None:
        return dash.no_update
    elif ctx.triggered_id == "treemap":
        # if treemap is clicked, hide/reveal the heatmap
        if last_heatmap_cls == "is-closed":
            heatmap_cls = "is-open"
        else:
            heatmap_cls = "is-closed"
    elif ctx.triggered_id in (
        "date-rangeslider",
        "project-dropdown",
        "component-dropdown",
    ):
        heatmap_cls = "is-closed"

    if heatmap_cls == "is-closed":
        heatmap = default_chart(bgcolor="#171717")
        treemap_cls = "treemap-subcharts-container"
        return heatmap, heatmap_cls, treemap_cls

    label = treemapClickData["points"][0]["label"]
    project_turbine, component_type = label.split("<br>")
    project = project_turbine.split("-")[0]
    turbine = project_turbine.split("-")[1]

    mean_col_map = treemap_data_from_file.columns.str.endswith("_mean")
    dff = treemap_data_from_file.copy()
    dff = dff.loc[:, ~mean_col_map]
    dff = format_columns(dff)

    lost_energy_df = None
    if heatmap_toggle == "by-component":
        # we show all components for selected turbine
        dff = filter_treemap_columns(
            data_frame=dff,
            project=project,
            turbine=turbine,
            component_type=None,
        )
        mean_df = filter_mean_values(treemap_data_from_file, project)

    elif heatmap_toggle == "by-turbine":
        # we show all turbines for selected component
        dff = filter_treemap_columns(
            data_frame=dff,
            project=project,
            turbine=None,
            component_type=component_type,
        )

        lost_energy_df = filter_lost_energy(
            treemap_data_from_file.loc[:, ~mean_col_map], project
        )
        mean_df = filter_mean_values(treemap_data_from_file, project, component_type)

    treemap_cls = "treemap-subcharts-container bottom-margin"
    heatmap = generate_comp_temp_heatmap(
        data_frame=dff, lost_energy_df=lost_energy_df, mean_frame=mean_df
    )

    return heatmap, heatmap_cls, treemap_cls


@callback(
    Output("neighbors-dropdown", "options"),
    Output("neighbors-dropdown", "value"),
    Output("power-perf-click-store", "data"),
    Input("pp-treemap", "clickData"),
    State("heatmap-box3", "className"),
)
def load_neighbors_dropdown(
    treemapClickData,
    subchart_cls,
):
    if not treemapClickData or subchart_cls == "is-open":
        return dash.no_update

    # figure out the name of the Turbine we clicked
    label = treemapClickData["points"][0]["label"]
    selected_target, component_type = label.split("<br>")

    surrogation_strategies = load_surrogation_strategies()
    top_neighbors = surrogation_strategies[
        surrogation_strategies["target"] == selected_target
    ]
    top_neighbors = top_neighbors.nlargest(10, "bulk_R2")
    options = [
        {
            "label": f"{row['surrogate']} (R²: {row['bulk_R2']:.2f})",
            "value": row["surrogate"],
        }
        for index, row in top_neighbors.iterrows()
    ]

    # automatically select the top 3 neighbors to display
    values = [option["value"] for option in options[:3]]
    return options, values, selected_target


@callback(
    Output("level-1-subplot", "figure"),
    Input("power-perf-click-store", "data"),
    State("neighbors-dropdown", "value"),
    State("start-date-clean2", "data"),
    State("end-date-clean2", "data"),
)
def load_level1_power_curve_subcharts(
    selected_target,
    selected_neighbors,
    start_date,
    end_date,
):
    # generate the level-1 pieces
    power_curve_df = load_power_curve_data()
    distribution_df = load_power_distribution_data()
    level_1_chart = gen_level1_subcharts(
        power_curve_df=power_curve_df,
        distribution_df=distribution_df,
        ws_distribution=ws_dist,
        turbine=selected_target,
        start_date=start_date,
        end_date=end_date,
    )
    return level_1_chart


@callback(
    Output("peer2peer-chart", "figure"),
    Output("peer-to-peer-table", "data"),
    Input("power-perf-click-store", "data"),
    Input("neighbors-dropdown", "value"),
    State("start-date-clean2", "data"),
    State("end-date-clean2", "data"),
)
def load_level2_power_curve_subcharts(
    selected_target,
    selected_neighbors,
    start_date,
    end_date,
):
    if selected_target in (None, []):
        return dash.no_update

    power_curve_df = load_power_curve_data()
    distribution_df = load_power_distribution_data()
    ws_dist = load_ws_distribution_data()
    surrogation_strategies = load_surrogation_strategies()

    power_curves = power_curve_df.loc[
        power_curve_df.index.get_level_values("Day") != "All"
    ]

    # Filter power curves based on selected date range and average over the days
    mask = (
        pd.to_datetime(power_curves.index.get_level_values("Day")) >= start_date
    ) & (pd.to_datetime(power_curves.index.get_level_values("Day")) <= end_date)

    filtered_power_curves = power_curves.loc[mask]
    filtered_distribution = distribution_df[mask]

    # check that the target turbine is in the frame
    if (selected_target not in filtered_power_curves.index) | (
        selected_target not in filtered_distribution.index
    ):
        print(
            f"performance.load_level2_power_curve_subcharts: target turbine {selected_target} not found in power curve or distribution data."
        )
        return dash.no_update

    all_power_curve_counts = {}
    avg_power_curves = {}
    all_r2_values = []
    for neighbor in selected_neighbors:
        r2_value = surrogation_strategies.loc[
            (surrogation_strategies["target"] == selected_target)
            & (surrogation_strategies["surrogate"] == neighbor),
            "bulk_R2",
        ].values[0]
        all_r2_values.append(r2_value)

        # make sure the turbines we need are there if not move to the next
        if (neighbor not in filtered_power_curves.index) or (
            neighbor not in filtered_distribution.index
        ):
            print(
                f"performance.load_level2_power_curve_subcharts: peer turbine {neighbor} not found in power curve or distribution data. Moving to the next turbine"
            )
            selected_neighbors = selected_neighbors.remove(neighbor)
            continue
        # aggregate the power curves by weighting by bin counts
        (
            turbine_power_curve,
            turbine_power_curve_counts,
        ) = calculate_weighted_average_power_by_turbine(
            filtered_power_curves.loc[neighbor], filtered_distribution.loc[neighbor]
        )

        avg_power_curves[neighbor] = turbine_power_curve
        all_power_curve_counts[neighbor] = turbine_power_curve_counts

    (
        turbine_power_curve,
        turbine_power_curve_counts,
    ) = calculate_weighted_average_power_by_turbine(
        filtered_power_curves.loc[selected_target],
        filtered_distribution.loc[selected_target],
    )
    avg_power_curves[selected_target] = turbine_power_curve
    all_power_curve_counts[selected_target] = turbine_power_curve_counts

    turbine_power_curves = pd.concat(avg_power_curves, axis=1).transpose()

    level_2_chart = gen_peer_to_peer_chart(
        selected_neighbors,
        selected_target,
        surrogation_strategies,
        turbine_power_curves,
    )

    TABLE_COLUMNS = ("Pairings", "% AEP Delta")
    TABLE_NULL_VAL = "-"
    table_data = []

    target_power_curve = filtered_power_curves.loc[selected_target].copy()
    target_power_curve_counts = filtered_distribution.loc[selected_target].copy()

    if selected_neighbors is not None:
        for neighbor in selected_neighbors:
            neighbor_power_curve = filtered_power_curves.loc[neighbor]
            neighbor_power_curve_distribution = filtered_distribution.loc[neighbor]

            common_indices = find_common_valid_indices(
                target_power_curve, neighbor_power_curve
            )

            target_aep, ____, ____ = full_calculate_AEP(
                project=selected_target.split("-")[0],
                power_curve=target_power_curve,
                power_curve_distribution=target_power_curve_counts,
                ws_dist=ws_dist,
                valid_indices=common_indices,
            )

            neighbor_aep, ___, ___ = full_calculate_AEP(
                project=selected_target.split("-")[0],
                power_curve=neighbor_power_curve,
                power_curve_distribution=neighbor_power_curve_distribution,
                ws_dist=ws_dist,
                valid_indices=common_indices,
            )
            aep_delta = round(((target_aep - neighbor_aep) / neighbor_aep) * 100, 1)

            row = {
                TABLE_COLUMNS[0]: f"{selected_target} → {neighbor}",
                TABLE_COLUMNS[1]: aep_delta,
            }
            table_data.append(row)

    return level_2_chart, table_data
