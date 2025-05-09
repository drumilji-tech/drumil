import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import PIconnect as PI
import plotly.express as px
from datetime import datetime
import pytz
import time

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# Function to fetch data from PI with retries and data cleaning
def fetch_data(server_name, tag_names, start_date, end_date, interval):
    max_retries = 3
    retry_delay = 10  # seconds
    attempt = 0
    
    while attempt < max_retries:
        try:
            with PI.PIServer(server_name) as server:
                data_frames = []
                for tag in tag_names:
                    points = server.search(tag)[0]
                    data = points.interpolated_values(start_date, end_date, interval)
                    df = pd.DataFrame({
                        'Time': data.index,
                        tag: pd.to_numeric(data.values, errors='coerce')
                    })
                    data_frames.append(df)
                merged_df = pd.concat(data_frames, axis=1).dropna()
                merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]  # Remove duplicate columns
                return merged_df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(retry_delay)
            attempt += 1
    
    return pd.DataFrame()  # return empty DataFrame if all retries fail

# Layout of the Dash app
app.layout = dbc.Container([
    html.H1("PI Data Plotting Tool", style={'textAlign': 'center'}),
    
    dbc.Row([
        dbc.Col([
            html.Label('PI Server Name'),
            dbc.Input(id='server_name', value='ALXAPAP680', type='text'),
            html.Br(),
            
            html.Label('X-axis Option'),
            dcc.Dropdown(
                id='x_axis_option',
                options=[
                    {'label': 'Timestamp', 'value': 'timestamp'},
                    {'label': 'Tag', 'value': 'tag'}
                ],
                value='timestamp'
            ),
            html.Br(),
            
            html.Label('X-axis Tag Name (if Tag chosen)'),
            dbc.Input(id='tag_x', value='', type='text'),
            html.Br(),
            
            html.Label('Y-axis Tag Names (comma separated for multiple tags)'),
            dbc.Input(id='tag_y', value='', type='text'),
            html.Br(),
            
            html.Label('Plot Type'),
            dcc.Dropdown(
                id='plot_type',
                options=[
                    {'label': '2D', 'value': '2d'},
                    {'label': '3D', 'value': '3d'}
                ],
                value='2d'
            ),
            html.Br(),
            
            html.Label('Z-axis Tag Name (if 3D chosen)'),
            dbc.Input(id='tag_z', value='', type='text'),
            html.Br(),
            
            html.Label('Start Date'),
            dcc.DatePickerSingle(
                id='start_date',
                date=datetime(2024, 1, 1)
            ),
            html.Br(),
            
            html.Label('End Date'),
            dcc.DatePickerSingle(
                id='end_date',
                date=datetime(2024, 2, 1)
            ),
            html.Br(),
            
            html.Label('Time Interval'),
            dcc.Dropdown(
                id='interval',
                options=[
                    {'label': '10 minutes', 'value': '10m'},
                    {'label': '15 minutes', 'value': '15m'},
                    {'label': '30 minutes', 'value': '30m'},
                    {'label': '1 hour', 'value': '1h'}
                ],
                value='15m'
            ),
            html.Br(),
            
            dbc.Button(id='submit-button', n_clicks=0, children='Submit', color='primary', className='w-100')
        ], width=4),
        
        dbc.Col([
            dcc.Loading(
                id="loading-graph",
                type="default",
                children=html.Div(id='output-graph', children="Please submit the form to see the graph.")
            )
        ], width=8)
    ])
], fluid=True)

# Callback to update the graph
@app.callback(
    Output('output-graph', 'children'),
    [
        Input('submit-button', 'n_clicks')
    ],
    [
        State('server_name', 'value'),
        State('x_axis_option', 'value'),
        State('tag_x', 'value'),
        State('tag_y', 'value'),
        State('tag_z', 'value'),
        State('plot_type', 'value'),
        State('start_date', 'date'),
        State('end_date', 'date'),
        State('interval', 'value')
    ]
)
def update_graph(n_clicks, server_name, x_axis_option, tag_x, tag_y, tag_z, plot_type, start_date, end_date, interval):
    if n_clicks > 0:
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M:%S%z')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d %H:%M:%S%z')
        
        tags_y = [tag.strip() for tag in tag_y.split(',')]
        
        if x_axis_option == 'timestamp':
            df = fetch_data(server_name, tags_y, start_date, end_date, interval)
            if df.empty:
                return html.Div("Failed to load data. Please check your inputs and try again.")
            
            if plot_type == '2d':
                fig = px.line(df, x='Time', y=tags_y, title='Timestamp vs Tags', labels={'value': 'Value', 'variable': 'Tag'}, width=1600, height=800)
            elif plot_type == '3d':
                df = df.melt(id_vars=['Time'], value_vars=tags_y, var_name='Tag', value_name='Value')
                fig = px.scatter_3d(df, x='Time', y='Tag', z='Value', title='3D Plot: Timestamp vs Tags', labels={'Value': 'Value', 'Tag': 'Tag'}, width=3200, height=1600)
        
        elif x_axis_option == 'tag':
            all_tags = [tag_x] + tags_y
            if plot_type == '3d':
                all_tags.append(tag_z)
            df = fetch_data(server_name, all_tags, start_date, end_date, interval)
            if df.empty:
                return html.Div("Failed to load data. Please check your inputs and try again.")
            
            if plot_type == '2d':
                fig = px.scatter(df, x=tag_x, y=tags_y, title=f'{tag_x} vs Tags', labels={'value': 'Value', 'variable': 'Tag'}, width=1600, height=800)
            elif plot_type == '3d':
                fig = px.scatter_3d(df, x=tag_x, y=tags_y[0], z=tag_z, title=f'3D Plot: {tag_x} vs {tags_y[0]} and {tag_z}', labels={'x': tag_x, 'y': tags_y[0], 'z': tag_z}, width=3200, height=1600)
        
        return dcc.Graph(figure=fig)
    
    return html.Div("Please submit the form to see the graph.")

if __name__ == '__main__':
    app.run_server(debug=True)
