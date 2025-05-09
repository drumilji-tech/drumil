from dash import Dash, html
app = Dash(__name__)
server = app.server          # Azure needs this

app.layout = html.H1("Hi Team! From Dash on Azure! Built by Drumil Joshi")
if __name__ == "__main__":
    app.run_server(debug=True)
