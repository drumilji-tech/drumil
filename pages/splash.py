"""The App's Landing Page with navigation to its contained Pages."""

import dash
import dash_bootstrap_components as dbc

dash.register_page(
    __name__,
    path="/",
    title="AI²",
)


def layout():
    return dbc.Row(
        className="g-0",
        justify="around",
        children=[
            dbc.Col(
                children=dbc.NavLink(
                    children="Performance & Reliability",
                    className="splash-page-card",
                    href=dash.get_relative_path("/performance-and-reliability"),
                ),
                sm=12,
                md=5,
            ),
            dbc.Col(
                children=dbc.NavLink(
                    children="Fault Analysis",
                    className="splash-page-card",
                    href=dash.get_relative_path("/fault-analysis"),
                ),
                sm=12,
                md=5,
            ),
        ],
    )
