"""The UI Components of the app."""

import math

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html, dash_table

from Utils.Constants import (
    DEFAULT_PARSE_FUNCS,
    OEM_PROJECT_MAPPING,
)
from Utils.Transformers import (
    get_projects,
)
from Utils.UiConstants import (
    HIDDEN_STYLE,
    APP_TITLE,
)


def gen_project_dropdown(treemap_data_from_file):
    project_list = get_projects(
        data=treemap_data_from_file, project_func=DEFAULT_PARSE_FUNCS["project_func"]
    )
    project_dropdown_options = [
        {"label": project, "value": project} for project in project_list
    ]
    project_dropdown = dcc.Dropdown(
        id="project-dropdown",
        className="dropdown-proj",
        options=project_dropdown_options,
        value="All",
        clearable=False,
    )
    return project_dropdown


def gen_date_intervals(treemap_data_from_file):
    """Create a sequence of dates for the date slider.

    Args:
        treemap_data_from_file (pandas.DataFrame): The
            treemap dataset.
    Returns:
        (pandas.DatetimeIndex): A Pandas DatetimeIndex.
    """

    slider_dates = pd.date_range(
        treemap_data_from_file.index[0],
        treemap_data_from_file.index[-1],
        freq="D",
    )
    
    date_intervals = pd.date_range(
        start=min(slider_dates), end=max(slider_dates), freq="1D"
    )
    
    return date_intervals


def gen_date_slider(date_intervals):
    """Build a Dash date slider with intelligently spaced marks.

    The date slider on a Macbook air is ~1400px wide,
    and width of our formatted datetime in our font is ~30px (eg. "02/07").

    The ratio of 1400px/30px = 46.677 is the value we use to benchmark and
    truncate the number of visible marks on the date slider.

    The variable `gap` contained in this function controls the number mark
    labels we skip displaying. If 0, every mark gets a label. If 1, every
    other mark gets a label, etc.

    Args:
        date_intervals (Pandas array): A Pandas array of date intervals.

    Returns:
        (dcc.RangeSlider): A Dash date slider.
    """
    updated_intervals = date_intervals
    gap = math.ceil(len(updated_intervals) / 46.677)
    marks = {}
    for i, interval in enumerate(updated_intervals):
        if i % gap == 0 or i == 0 or i == len(updated_intervals) - 1:
            marks[i] = {"label": interval.strftime("%m/%d")}
        else:
            marks[i] = {"label": ""}

    date_slider = dcc.RangeSlider(
        id="date-rangeslider",
        className="date-rangeslider",
        min=0,
        max=len(updated_intervals) - 1,
        step=1,
        value=[(len(updated_intervals) - 8) if len(updated_intervals) > 7 else 0, len(updated_intervals) - 1],
        marks=marks,
        pushable=1,
        included=True,
    )
    return date_slider


def acknowledge_control(
    _id, label=None, placeholder=None, as_children=None, fluid_width=None
):
    """A widget that is meant to filter out data in other charts.

    Sometimes when you are looking at a chart and see visual pieces that
    stem from one attribute in the underlying dataset, it can clog up the
    chart, hiding other attributes from plain sight.

    This component is part of a system that, when hooked up, can allow you
    to stop seeing particular attributes so you can bring those hidden
    attributes front and center.

    In this application, the attribute that we don't want to see or NOT see
    are the Fault Codes.
        If one Fault Code were to contribute to 90% of the tiles in a treemap,
        we would like to easily hide or 'acknowledge' those tiles and see the
        contributes of the other Fault Codes.

    By populating this widget's dropdown with all the values of a given
    attribute, and by hooking it up to our desired charts, we can improve
    the user's analysis of the underlying data.

    Args:
        _id (str): The id of the `dcc.Dropdown` in this control.
        label (str): The label of the control.
        placeholder (str): The placholder string that appears in the dropdown.
        as_children (bool): See the Returns section.
        fluid_width (bool): If True, the dropdown will fill up the width
            of its parent. If False, it will be fixed as a wide dropdown.

    Returns:
        (html.Div): The control with a label and dropdown. The `as_children`
            param determines if the output is a raw list of the components
            (True) or whether it is wrapped in an `html.Div` (False).
    """
    if label is None:
        label = "Turbines to Hide"
    if placeholder is None:
        placeholder = "Add Turbine Codes to hide from the treemap view..."
    if as_children is None:
        as_children = False
    if fluid_width is None or fluid_width is False:
        dropdown_cls = "acknowledged-pp-turbines"
    else:
        dropdown_cls = ""

    children = [
        html.Label(
            className="dropdown-label",
            htmlFor="acknowledged-pp-turbines",
            children=label,
        ),
        dcc.Dropdown(
            id=_id,
            className=dropdown_cls,
            value=[],
            placeholder=placeholder,
            multi=True,
            persistence=True,
            persistence_type="local",
        ),
    ]
    if as_children:
        return children
    return html.Div(children)


def gen_table_component(_id, table_columns):
    """Make a AEP table that sits inside a chart.

    The table only appears to float inside the chart because
    of the magic of CSS.

    Args:
        _id (str): The unique ID of the dash_table.DataTable contained.
    Returns:
        (html.Div): A Div that contains a dash table.

    """
    cell_height = 16
    component = html.Div(
        children=dash_table.DataTable(
            id=_id,
            columns=[{"id": c, "name": c} for c in table_columns],
            style_header={
                "backgroundColor": "rgb(30, 30, 30)",
                "color": "white",
            },
            style_data={
                "backgroundColor": "rgb(50, 50, 50)",
                "color": "white",
            },
            style_cell={
                "textAlign": "center",
                "height": f"{cell_height}px",
            },
        ),
    )
    return component


def gen_sticky_header(treemap_data_from_file, date_intervals, registry_values):
    """Generate the app's header with logo, title, and controls.

    The first thing you see when visit this app is the header
    at the top of the page. This header is special because it
    stays fixed and hovering above the rest of the app as you scroll
    up and down the page.

    The reason why this header sticks to the top is because it contains
    global controls (eg. dropdown(s)) that affect all charts in the app.
    You may want to scroll down with the global controls in sights so
    you can conveniently make change in the control and see your chart
    update in real time.

    Args:
        treemap_data_from_file (pandas.DataFrame): This is the data that
            is used to populate the treemap chart. This param should
            come from the output of the `load_treemap_dataset` helper.
        date_intervals (pandas.DateRange): A series of dates that are used
            to generate a time date ranger slider in the header.
        registry_values (odict_values): Should be equal to the output of
            `dash.page_registry.values()`. This param is used to setup
            navigation to helper users move between different pages of
            this multi-page app.

    Returns:
        (dash.html.Div Component): A UI component for the app.

    Raises:
        None
    """
    project_dropdown = gen_project_dropdown(treemap_data_from_file)
    date_slider = gen_date_slider(date_intervals)

    nav_contents = []
    for page in registry_values:
        if page["title"] != "AI²":
            _id = page["title"].replace(" ", "-").lower()
            navitem = dbc.NavItem(
                dbc.NavLink(
                    id=_id,
                    children=page["title"],
                    href=page["relative_path"],
                    active="exact",
                ),
            )
            nav_contents.append(navitem)
    return html.Div(
        className="preheader",
        children=[
            dcc.Location(id="url"),
            dbc.Row(
                className="g-0",
                align="center",
                justify="between",
                children=[
                    dbc.Col(
                        children=[
                            dcc.Link(
                                children=html.Img(
                                    style={
                                        "height": "1.65rem",
                                        "transform": "translateY(3px)",
                                    },
                                    src=dash.get_asset_url(
                                        "spc_user_logo_transparent.png"
                                    ),
                                ),
                                href=dash.get_relative_path("/"),
                            ),
                        ],
                        md=1,
                        style={"width": "10px"},
                    ),
                    dbc.Col(
                        dbc.NavbarBrand(APP_TITLE, className="ms-2"),
                        md=7,
                    ),
                    dbc.Col(
                        id="nav-item-1",
                        style=HIDDEN_STYLE,
                        md=2,
                        children=nav_contents[1],
                    ),
                    dbc.Col(
                        id="nav-item-2",
                        style=HIDDEN_STYLE,
                        md=2,
                        children=nav_contents[0],
                    ),
                ],
            ),
            dbc.Row(
                id="header-first-row",
                className="g-2",
                align="center",
                justify="between",
                children=[
                    dbc.Col(
                        id="trip-notrip-radioitems-box",
                        children=[
                            dcc.Checklist(
                                id="trip-notrip-radioitems",
                                style={"lineHeight": "34px"},
                                options=["Trip", "No Trip"],
                                value=["Trip", "No Trip"],
                                labelStyle={"display": "inline-block"},
                            ),
                        ],
                        className="disabled",
                        width=1,
                    ),
                    dbc.Col(
                        id="hot-cold-radioitems-box",
                        children=[
                            dcc.Checklist(
                                id="hot-cold-radioitems",
                                style={"lineHeight": "34px"},
                                options=["Hot", "Cold"],
                                value=["Hot", "Cold"],
                                labelStyle={"display": "inline-block"},
                            ),
                        ],
                        className="disabled",
                        width=1,
                    ),
                    dbc.Col(
                        id="filter-faults-box",
                        children=[
                            html.Label(
                                className="filter-faults-label",
                                htmlFor="filter-faults-dropdown",
                                children="Filter by Faults",
                            ),
                            dcc.Dropdown(
                                id="filter-faults-dropdown",
                                options=["100", "101", "102", "103"],
                                multi=True,
                                clearable=False,
                                placeholder="Filter by Faults...",
                            ),
                        ],
                        className="disabled",
                        width=2,
                    ),
                    dbc.Col(
                        id="acknowledged-fault-descriptions-box",
                        children=[
                            *acknowledge_control(
                                _id="acknowledged-fault-descriptions",
                                label="Acknowledge Faults",
                                placeholder="Type to find faults...",
                                as_children=True,
                                fluid_width=True,
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        id="oem-dropdown-box",
                        children=[
                            html.Label(
                                className="oem-dropdown-label",
                                htmlFor="oem-dropdown",
                                children="Filter by OEM",
                            ),
                            dcc.Dropdown(
                                id="oem-dropdown",
                                options=["All"] + list(OEM_PROJECT_MAPPING.keys()),
                                value="All",
                                clearable=False,
                            ),
                        ],
                        className="",
                        width=2,
                    ),
                    dbc.Col(
                        children=[
                            html.Label(
                                className="project-dropdown-label",
                                htmlFor="project-dropdown",
                                children="Filter By Project",
                            ),
                            project_dropdown,
                        ],
                        width=2,
                    ),
                ],
            ),
            html.Div(
                id="date-selector-box",
                className="card control-panel",
                children=[
                    html.Div(children=date_slider, className="date-rangeslider-box"),
                ],
            ),
        ],
    )
