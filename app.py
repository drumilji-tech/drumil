import dash
import dash_bootstrap_components as dbc
from dash import html   # just for a placeholder layout


app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="AI Dashboard",
    use_pages=True,
    external_stylesheets=[dbc.themes.GRID],
)


server = app.server     # ← THIS lets Gunicorn find the Flask server


# quick smoke-test layout (you can delete once real pages load)
app.layout = html.Div(
    style={"textAlign": "center", "marginTop": "4rem"},
    children=html.H2("✅ Azure deployment reached Dash app!")
)


app.scripts.config.serve_locally = True
app.css.config.serve_locally     = True


if __name__ == "__main__":       # so it only runs locally, not under Gunicorn
    app.run_server(debug=True)
