import dash
import dash_bootstrap_components as dbc
from dash import dcc, Input, Output

from app import app
from Utils.Components import (
    gen_date_intervals,
    gen_sticky_header,
)
from Utils.Loaders import (
    load_treemap_dataset,
)
from Utils.UiConstants import (
    HIDDEN_STYLE,
    VISIBLE_STYLE,
)

treemap_data_from_file = load_treemap_dataset()


def entry_layout():
    date_intervals = gen_date_intervals(treemap_data_from_file)
    sticky_header = gen_sticky_header(
        treemap_data_from_file=treemap_data_from_file,
        date_intervals=date_intervals,
        registry_values=dash.page_registry.values(),
    )
    return dbc.Container(
        id="main-container",
        fluid=True,
        class_name="wrap",
        children=[
            dcc.Store(
                id="date-intervals-store",
                data=date_intervals,
            ),
            sticky_header,
            dash.page_container,
        ],
    )


app.layout = entry_layout()


@app.callback(
    Output("header-first-row", "style"),
    Output("date-selector-box", "style"),
    Output("main-container", "style"),
    Output("nav-item-1", "style"),
    Output("nav-item-2", "style"),
    Input("url", "pathname"),
)
def toggle_global_controls(pathname):
    parsed_pathname = dash.strip_relative_path(pathname)
    if not parsed_pathname:  # None or ""
        return (
            HIDDEN_STYLE,
            HIDDEN_STYLE,
            {"background-color": "rgba(0,0,0,0)"},
            HIDDEN_STYLE,
            HIDDEN_STYLE,
        )
    return (
        VISIBLE_STYLE,
        VISIBLE_STYLE,
        {"background-color": "#171717"},
        VISIBLE_STYLE,
        VISIBLE_STYLE,
    )


@app.callback(
    Output("trip-notrip-radioitems-box", "style"),
    Output("hot-cold-radioitems-box", "style"),
    Output("filter-faults-box", "style"),
    Output("acknowledged-fault-descriptions-box", "style"),
    Output("oem-dropdown-box", "style"),
    Input("url", "pathname"),
)
def toggle_fault_controls(pathname):
    parsed_pathname = dash.strip_relative_path(pathname)
    if parsed_pathname == "fault-analysis":
        return [VISIBLE_STYLE] * 5
    else:
        return [HIDDEN_STYLE] * 5


if __name__ == "__main__":
    app.run_server(debug=True)
