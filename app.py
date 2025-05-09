from dash import Dash, html
app = Dash(__name__)
server = app.server          # Azure needs this

app.layout = html.H1("Hello from Dash on Azure!")
if __name__ == "__main__":
    app.run_server(debug=True)
