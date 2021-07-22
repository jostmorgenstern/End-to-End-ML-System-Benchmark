import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output
from e2ebench import VisualizationBenchmark
from e2ebench.visualization import type_to_visualizer_class_mapper

external_stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheet)
server = app.server


def get_all_uuids(df):
    uuid_choices = df['uuid'].to_list()
    uuid_choices = list(dict.fromkeys(uuid_choices))
    return uuid_choices


def get_all_types(df):
    type_choices = df['measurement_type'].to_list()
    type_choices = list(dict.fromkeys(type_choices))
    return type_choices


def get_all_descriptions(df):
    desc_choices = df['measurement_description'].to_list()
    desc_choices = list(dict.fromkeys(desc_choices))
    return desc_choices


app.layout = html.Div(
    children=[
        html.H1(children='Umlaut'),

        html.H3('Figure 1'),
        dcc.Input(id='db_file', type='text', placeholder='Enter the path to your database file here',
                  size='50',
                  debounce=True),

        html.Div(id='db_values'),

        html.Div([
            html.Label('UUID'),
            dcc.Dropdown(
                id='uuid_dropdown',
            ),
        ]),
        html.Div([
            html.Label('Type'),
            dcc.Dropdown(
                id='type_dropdown',
            ),
        ]),
        html.Div([
            html.Label('Description'),
            dcc.Dropdown(
                id='desc_dropdown',
            ),
        ]),
        html.Br(),
        html.Div(id='graph_1'),
        html.Br(),
        html.Button('Show/ Hide Section 2', id='add_button_2', n_clicks=0),
        html.Div(
            id='section_2',
            children=[
                html.H3('Figure 2'),
                html.Div([
                    html.Label('UUID'),
                    dcc.Dropdown(
                        id='uuid_dropdown_2',
                    ),
                ]),
                html.Div([
                    html.Label('Type'),
                    dcc.Dropdown(
                        id='type_dropdown_2',
                    ),
                ]),
                html.Div([
                    html.Label('Description'),
                    dcc.Dropdown(
                        id='desc_dropdown_2',
                    ),
                ]),
                html.Br(),
                html.Div(id='graph_2'),
                html.Br(),
                html.Button('Show/ Hide Section 3', id='add_button_3', n_clicks=0),
                html.Div(
                    id='section_3',
                    children=[
                        html.H3('Figure 3'),
                        html.Div([
                            html.Label('UUID'),
                            dcc.Dropdown(
                                id='uuid_dropdown_3',
                            ),
                        ]),
                        html.Div([
                            html.Label('Type'),
                            dcc.Dropdown(
                                id='type_dropdown_3',
                            ),
                        ]),
                        html.Div([
                            html.Label('Description'),
                            dcc.Dropdown(
                                id='desc_dropdown_3',
                            ),
                        ]),
                        html.Br(),
                        html.Div(id='graph_3'),
                        html.Br(),
                        html.Button('Show/ Hide Section 4', id='add_button_4', n_clicks=0),
                        html.Div(
                            id='section_4',
                            children=[
                                html.H3('Figure 4'),
                                html.Div([
                                    html.Label('UUID'),
                                    dcc.Dropdown(
                                        id='uuid_dropdown_4',
                                    ),
                                ]),
                                html.Div([
                                    html.Label('Type'),
                                    dcc.Dropdown(
                                        id='type_dropdown_4',
                                    ),
                                ]),
                                html.Div([
                                    html.Label('Description'),
                                    dcc.Dropdown(
                                        id='desc_dropdown_4',
                                    ),
                                ]),
                                html.Br(),
                                html.Div(id='graph_4'),
                                html.Br(),
                                html.Button('Show/ Hide Section 5', id='add_button_5', n_clicks=0),
                                html.Div(
                                    id='section_5',
                                    children=[
                                        html.H3('Figure 5'),
                                        html.Div([
                                            html.Label('UUID'),
                                            dcc.Dropdown(
                                                id='uuid_dropdown_5',
                                            ),
                                        ]),
                                        html.Div([
                                            html.Label('Type'),
                                            dcc.Dropdown(
                                                id='type_dropdown_5',
                                            ),
                                        ]),
                                        html.Div([
                                            html.Label('Description'),
                                            dcc.Dropdown(
                                                id='desc_dropdown_5',
                                            ),
                                        ]),
                                        html.Br(),
                                        html.Div(id='graph_5'),
                                        html.Br(),
                                        html.Button('Show/ Hide Section 6', id='add_button_6', n_clicks=0),
                                        html.Div(
                                            id='section_6',
                                            children=[
                                                html.H3('Figure 6'),
                                                html.Div([
                                                    html.Label('UUID'),
                                                    dcc.Dropdown(
                                                        id='uuid_dropdown_6',
                                                    ),
                                                ]),
                                                html.Div([
                                                    html.Label('Type'),
                                                    dcc.Dropdown(
                                                        id='type_dropdown_6',
                                                    ),
                                                ]),
                                                html.Div([
                                                    html.Label('Description'),
                                                    dcc.Dropdown(
                                                        id='desc_dropdown_6',
                                                    ),
                                                ]),
                                                html.Br(),
                                                html.Div(id='graph_6'),
                                                html.Br(),
                                                html.Button('Show/ Hide Section 7', id='add_button_7', n_clicks=0),
                                                html.Div(
                                                    id='section_7',
                                                    children=[
                                                        html.H3('Figure 7'),
                                                        html.Div([
                                                            html.Label('UUID'),
                                                            dcc.Dropdown(
                                                                id='uuid_dropdown_7',
                                                            ),
                                                        ]),
                                                        html.Div([
                                                            html.Label('Type'),
                                                            dcc.Dropdown(
                                                                id='type_dropdown_7',
                                                            ),
                                                        ]),
                                                        html.Div([
                                                            html.Label('Description'),
                                                            dcc.Dropdown(
                                                                id='desc_dropdown_7',
                                                            ),
                                                        ]),
                                                        html.Br(),
                                                        html.Div(id='graph_7'),
                                                        html.Br(),
                                                        html.Button('Show/ Hide Section 8', id='add_button_8',
                                                                    n_clicks=0),
                                                        html.Div(
                                                            id='section_8',
                                                            children=[
                                                                html.H3('Figure 8'),
                                                                html.Div([
                                                                    html.Label('UUID'),
                                                                    dcc.Dropdown(
                                                                        id='uuid_dropdown_8',
                                                                    ),
                                                                ]),
                                                                html.Div([
                                                                    html.Label('Type'),
                                                                    dcc.Dropdown(
                                                                        id='type_dropdown_8',
                                                                    ),
                                                                ]),
                                                                html.Div([
                                                                    html.Label('Description'),
                                                                    dcc.Dropdown(
                                                                        id='desc_dropdown_8',
                                                                    ),
                                                                ]),
                                                                html.Br(),
                                                                html.Div(id='graph_8'),
                                                                html.Br(),
                                                                html.Button('Show/ Hide Section 9', id='add_button_9',
                                                                            n_clicks=0),
                                                                html.Div(
                                                                    id='section_9',
                                                                    children=[
                                                                        html.H3('Figure 9'),
                                                                        html.Div([
                                                                            html.Label('UUID'),
                                                                            dcc.Dropdown(
                                                                                id='uuid_dropdown_9',
                                                                            ),
                                                                        ]),
                                                                        html.Div([
                                                                            html.Label('Type'),
                                                                            dcc.Dropdown(
                                                                                id='type_dropdown_9',
                                                                            ),
                                                                        ]),
                                                                        html.Div([
                                                                            html.Label('Description'),
                                                                            dcc.Dropdown(
                                                                                id='desc_dropdown_9',
                                                                            ),
                                                                        ]),
                                                                        html.Br(),
                                                                        html.Div(id='graph_9'),
                                                                        html.Br(),
                                                                        html.Button('Show/ Hide Section 10',
                                                                                    id='add_button_10', n_clicks=0),
                                                                        html.Div(
                                                                            id='section_10',
                                                                            children=[
                                                                                html.H3('Figure 10'),
                                                                                html.Div([
                                                                                    html.Label('UUID'),
                                                                                    dcc.Dropdown(
                                                                                        id='uuid_dropdown_10',
                                                                                    ),
                                                                                ]),
                                                                                html.Div([
                                                                                    html.Label('Type'),
                                                                                    dcc.Dropdown(
                                                                                        id='type_dropdown_10',
                                                                                    ),
                                                                                ]),
                                                                                html.Div([
                                                                                    html.Label('Description'),
                                                                                    dcc.Dropdown(
                                                                                        id='desc_dropdown_10',
                                                                                    ),
                                                                                ]),
                                                                                html.Br(),
                                                                                html.Div(id='graph_10'),
                                                                                html.Br(),
                                                                            ],
                                                                            hidden=False)
                                                                    ],
                                                                    hidden=False),
                                                            ],
                                                            hidden=False),
                                                    ],
                                                    hidden=False),
                                            ],
                                            hidden=False),
                                    ],
                                    hidden=False),
                            ],
                            hidden=False),
                    ],
                    hidden=False),
            ],
            hidden=False),
    ])


@app.callback(
    Output('section_2', 'hidden'),
    Input('add_button_2', 'n_clicks'),
    Input('section_2', 'hidden')
)
def enable_sec_2(clicks, current_state):
    if current_state:
        return False
    else:
        return True


@app.callback(
    Output('section_3', 'hidden'),
    Input('add_button_3', 'n_clicks'),
    Input('section_3', 'hidden')
)
def enable_sec_3(clicks, current_state):
    if current_state:
        return False
    else:
        return True


@app.callback(
    Output('section_4', 'hidden'),
    Input('add_button_4', 'n_clicks'),
    Input('section_4', 'hidden')
)
def enable_sec_4(clicks, current_state):
    if current_state:
        return False
    else:
        return True


@app.callback(
    Output('section_5', 'hidden'),
    Input('add_button_5', 'n_clicks'),
    Input('section_5', 'hidden')
)
def enable_sec_5(clicks, current_state):
    if current_state:
        return False
    else:
        return True


@app.callback(
    Output('section_6', 'hidden'),
    Input('add_button_6', 'n_clicks'),
    Input('section_6', 'hidden')
)
def enable_sec_6(clicks, current_state):
    if current_state:
        return False
    else:
        return True


@app.callback(
    Output('section_7', 'hidden'),
    Input('add_button_7', 'n_clicks'),
    Input('section_7', 'hidden')
)
def enable_sec_7(clicks, current_state):
    if current_state:
        return False
    else:
        return True


@app.callback(
    Output('section_8', 'hidden'),
    Input('add_button_8', 'n_clicks'),
    Input('section_8', 'hidden')
)
def enable_sec_8(clicks, current_state):
    if current_state:
        return False
    else:
        return True


@app.callback(
    Output('section_9', 'hidden'),
    Input('add_button_9', 'n_clicks'),
    Input('section_9', 'hidden')
)
def enable_sec_9(clicks, current_state):
    if current_state:
        return False
    else:
        return True


@app.callback(
    Output('section_10', 'hidden'),
    Input('add_button_10', 'n_clicks'),
    Input('section_10', 'hidden')
)
def enable_sec_10(clicks, current_state):
    if current_state:
        return False
    else:
        return True


# 1
@app.callback(
    Output('uuid_dropdown', 'options'),
    Input('db_file', 'value')
)
def filter_uuids(value):
    try:
        type_benchmark = VisualizationBenchmark(value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('type_dropdown', 'options'),
    Input('uuid_dropdown', 'value'),
    Input('db_file', 'value')
)
def filter_types(value, db_value):
    try:
        value_list = [value]
        type_benchmark = VisualizationBenchmark(db_value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        type_df = type_df[type_df['uuid'].isin(value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('desc_dropdown', 'options'),
    Input('uuid_dropdown', 'value'),
    Input('type_dropdown', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions(uuid_value, type_value, db_value):
    try:
        uuid_value_list = [uuid_value]
        type_value_list = [type_value]
        desc_benchmark = VisualizationBenchmark(db_value)
        desc_df = desc_benchmark.query_all_uuid_type_desc()
        desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
        desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
        desc_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('graph_1', 'children'),
    Input('uuid_dropdown', 'value'),
    Input('type_dropdown', 'value'),
    Input('desc_dropdown', 'value'),
    Input('db_file', 'value')
)
def update_graph_1(uuid_value, type_value, desc_value, db_value):
    try:
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        updated_benchmark = VisualizationBenchmark(db_value)
        updated_meas_df = updated_benchmark.query_all_uuid_type_desc()
        updated_meta_df = updated_benchmark.query_all_meta()

        if updated_uuids is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['uuid'].isin(updated_uuids)]
            updated_meta_df = updated_meta_df[updated_meta_df.index.isin(updated_uuids)]
        if updated_types is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_type'].isin(updated_types)]
        if updated_descriptions is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_description'].isin(updated_descriptions)]

        if updated_meas_df.empty:
            raise Exception("There are no database entries with the given uuids, types and descriptions.")

        updated_df = updated_benchmark.join_visualization_queries(updated_meas_df)

        updated_VisualizerClass = type_to_visualizer_class_mapper[updated_types[0]]
        updated_visualizer = updated_VisualizerClass(updated_df, 'plotly')
        updated_fig = updated_visualizer.plot_with_plotly()[0]

        updated_benchmark.close()

        return [dcc.Graph(id='fig_1', figure=updated_fig)]
    except:
        return ['No figure available']


# 2
@app.callback(
    Output('uuid_dropdown_2', 'options'),
    Input('db_file', 'value')
)
def filter_uuids_2(value):
    try:
        type_benchmark = VisualizationBenchmark(value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('type_dropdown_2', 'options'),
    Input('uuid_dropdown_2', 'value'),
    Input('db_file', 'value')
)
def filter_types_2(value, db_value):
    try:
        value_list = [value]
        type_benchmark = VisualizationBenchmark(db_value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        type_df = type_df[type_df['uuid'].isin(value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('desc_dropdown_2', 'options'),
    Input('uuid_dropdown_2', 'value'),
    Input('type_dropdown_2', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions_2(uuid_value, type_value, db_value):
    try:
        uuid_value_list = [uuid_value]
        type_value_list = [type_value]
        desc_benchmark = VisualizationBenchmark(db_value)
        desc_df = desc_benchmark.query_all_uuid_type_desc()
        desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
        desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
        desc_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('graph_2', 'children'),
    Input('uuid_dropdown_2', 'value'),
    Input('type_dropdown_2', 'value'),
    Input('desc_dropdown_2', 'value'),
    Input('db_file', 'value')
)
def update_graph_2(uuid_value, type_value, desc_value, db_value):
    try:
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        updated_benchmark = VisualizationBenchmark(db_value)
        updated_meas_df = updated_benchmark.query_all_uuid_type_desc()
        updated_meta_df = updated_benchmark.query_all_meta()

        if updated_uuids is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['uuid'].isin(updated_uuids)]
            updated_meta_df = updated_meta_df[updated_meta_df.index.isin(updated_uuids)]
        if updated_types is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_type'].isin(updated_types)]
        if updated_descriptions is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_description'].isin(updated_descriptions)]

        if updated_meas_df.empty:
            raise Exception("There are no database entries with the given uuids, types and descriptions.")

        updated_df = updated_benchmark.join_visualization_queries(updated_meas_df)

        updated_VisualizerClass = type_to_visualizer_class_mapper[updated_types[0]]
        updated_visualizer = updated_VisualizerClass(updated_df, 'plotly')
        updated_fig = updated_visualizer.plot_with_plotly()[0]

        updated_benchmark.close()

        return [dcc.Graph(id='fig_2', figure=updated_fig)]
    except:
        return ['No figure available']


# 3
@app.callback(
    Output('uuid_dropdown_3', 'options'),
    Input('db_file', 'value')
)
def filter_uuids_3(value):
    try:
        type_benchmark = VisualizationBenchmark(value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('type_dropdown_3', 'options'),
    Input('uuid_dropdown_3', 'value'),
    Input('db_file', 'value')
)
def filter_types_3(value, db_value):
    try:
        value_list = [value]
        type_benchmark = VisualizationBenchmark(db_value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        type_df = type_df[type_df['uuid'].isin(value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('desc_dropdown_3', 'options'),
    Input('uuid_dropdown_3', 'value'),
    Input('type_dropdown_3', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions_3(uuid_value, type_value, db_value):
    try:
        uuid_value_list = [uuid_value]
        type_value_list = [type_value]
        desc_benchmark = VisualizationBenchmark(db_value)
        desc_df = desc_benchmark.query_all_uuid_type_desc()
        desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
        desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
        desc_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('graph_3', 'children'),
    Input('uuid_dropdown_3', 'value'),
    Input('type_dropdown_3', 'value'),
    Input('desc_dropdown_3', 'value'),
    Input('db_file', 'value')
)
def update_graph_3(uuid_value, type_value, desc_value, db_value):
    try:
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        updated_benchmark = VisualizationBenchmark(db_value)
        updated_meas_df = updated_benchmark.query_all_uuid_type_desc()
        updated_meta_df = updated_benchmark.query_all_meta()

        if updated_uuids is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['uuid'].isin(updated_uuids)]
            updated_meta_df = updated_meta_df[updated_meta_df.index.isin(updated_uuids)]
        if updated_types is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_type'].isin(updated_types)]
        if updated_descriptions is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_description'].isin(updated_descriptions)]

        if updated_meas_df.empty:
            raise Exception("There are no database entries with the given uuids, types and descriptions.")

        updated_df = updated_benchmark.join_visualization_queries(updated_meas_df)

        updated_VisualizerClass = type_to_visualizer_class_mapper[updated_types[0]]
        updated_visualizer = updated_VisualizerClass(updated_df, 'plotly')
        updated_fig = updated_visualizer.plot_with_plotly()[0]

        updated_benchmark.close()

        return [dcc.Graph(id='fig_3', figure=updated_fig)]
    except:
        return ['No figure available']


# 4
@app.callback(
    Output('uuid_dropdown_4', 'options'),
    Input('db_file', 'value')
)
def filter_uuids_4(value):
    try:
        type_benchmark = VisualizationBenchmark(value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('type_dropdown_4', 'options'),
    Input('uuid_dropdown_4', 'value'),
    Input('db_file', 'value')
)
def filter_types_4(value, db_value):
    try:
        value_list = [value]
        type_benchmark = VisualizationBenchmark(db_value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        type_df = type_df[type_df['uuid'].isin(value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('desc_dropdown_4', 'options'),
    Input('uuid_dropdown_4', 'value'),
    Input('type_dropdown_4', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions_4(uuid_value, type_value, db_value):
    try:
        uuid_value_list = [uuid_value]
        type_value_list = [type_value]
        desc_benchmark = VisualizationBenchmark(db_value)
        desc_df = desc_benchmark.query_all_uuid_type_desc()
        desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
        desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
        desc_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('graph_4', 'children'),
    Input('uuid_dropdown_4', 'value'),
    Input('type_dropdown_4', 'value'),
    Input('desc_dropdown_4', 'value'),
    Input('db_file', 'value')
)
def update_graph_4(uuid_value, type_value, desc_value, db_value):
    try:
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        updated_benchmark = VisualizationBenchmark(db_value)
        updated_meas_df = updated_benchmark.query_all_uuid_type_desc()
        updated_meta_df = updated_benchmark.query_all_meta()

        if updated_uuids is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['uuid'].isin(updated_uuids)]
            updated_meta_df = updated_meta_df[updated_meta_df.index.isin(updated_uuids)]
        if updated_types is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_type'].isin(updated_types)]
        if updated_descriptions is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_description'].isin(updated_descriptions)]

        if updated_meas_df.empty:
            raise Exception("There are no database entries with the given uuids, types and descriptions.")

        updated_df = updated_benchmark.join_visualization_queries(updated_meas_df)

        updated_VisualizerClass = type_to_visualizer_class_mapper[updated_types[0]]
        updated_visualizer = updated_VisualizerClass(updated_df, 'plotly')
        updated_fig = updated_visualizer.plot_with_plotly()[0]

        updated_benchmark.close()

        return [dcc.Graph(id='fig_4', figure=updated_fig)]
    except:
        return ['No figure available']


# 5
@app.callback(
    Output('uuid_dropdown_5', 'options'),
    Input('db_file', 'value')
)
def filter_uuids_5(value):
    try:
        type_benchmark = VisualizationBenchmark(value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('type_dropdown_5', 'options'),
    Input('uuid_dropdown_5', 'value'),
    Input('db_file', 'value')
)
def filter_types_5(value, db_value):
    try:
        value_list = [value]
        type_benchmark = VisualizationBenchmark(db_value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        type_df = type_df[type_df['uuid'].isin(value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('desc_dropdown_5', 'options'),
    Input('uuid_dropdown_5', 'value'),
    Input('type_dropdown_5', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions_5(uuid_value, type_value, db_value):
    try:
        uuid_value_list = [uuid_value]
        type_value_list = [type_value]
        desc_benchmark = VisualizationBenchmark(db_value)
        desc_df = desc_benchmark.query_all_uuid_type_desc()
        desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
        desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
        desc_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('graph_5', 'children'),
    Input('uuid_dropdown_5', 'value'),
    Input('type_dropdown_5', 'value'),
    Input('desc_dropdown_5', 'value'),
    Input('db_file', 'value')
)
def update_graph_5(uuid_value, type_value, desc_value, db_value):
    try:
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        updated_benchmark = VisualizationBenchmark(db_value)
        updated_meas_df = updated_benchmark.query_all_uuid_type_desc()
        updated_meta_df = updated_benchmark.query_all_meta()

        if updated_uuids is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['uuid'].isin(updated_uuids)]
            updated_meta_df = updated_meta_df[updated_meta_df.index.isin(updated_uuids)]
        if updated_types is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_type'].isin(updated_types)]
        if updated_descriptions is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_description'].isin(updated_descriptions)]

        if updated_meas_df.empty:
            raise Exception("There are no database entries with the given uuids, types and descriptions.")

        updated_df = updated_benchmark.join_visualization_queries(updated_meas_df)

        updated_VisualizerClass = type_to_visualizer_class_mapper[updated_types[0]]
        updated_visualizer = updated_VisualizerClass(updated_df, 'plotly')
        updated_fig = updated_visualizer.plot_with_plotly()[0]

        updated_benchmark.close()

        return [dcc.Graph(id='fig_5', figure=updated_fig)]
    except:
        return ['No figure available']


# 6
@app.callback(
    Output('uuid_dropdown_6', 'options'),
    Input('db_file', 'value')
)
def filter_uuids_6(value):
    try:
        type_benchmark = VisualizationBenchmark(value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('type_dropdown_6', 'options'),
    Input('uuid_dropdown_6', 'value'),
    Input('db_file', 'value')
)
def filter_types_6(value, db_value):
    try:
        value_list = [value]
        type_benchmark = VisualizationBenchmark(db_value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        type_df = type_df[type_df['uuid'].isin(value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('desc_dropdown_6', 'options'),
    Input('uuid_dropdown_6', 'value'),
    Input('type_dropdown_6', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions_6(uuid_value, type_value, db_value):
    try:
        uuid_value_list = [uuid_value]
        type_value_list = [type_value]
        desc_benchmark = VisualizationBenchmark(db_value)
        desc_df = desc_benchmark.query_all_uuid_type_desc()
        desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
        desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
        desc_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('graph_6', 'children'),
    Input('uuid_dropdown_6', 'value'),
    Input('type_dropdown_6', 'value'),
    Input('desc_dropdown_6', 'value'),
    Input('db_file', 'value')
)
def update_graph_6(uuid_value, type_value, desc_value, db_value):
    try:
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        updated_benchmark = VisualizationBenchmark(db_value)
        updated_meas_df = updated_benchmark.query_all_uuid_type_desc()
        updated_meta_df = updated_benchmark.query_all_meta()

        if updated_uuids is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['uuid'].isin(updated_uuids)]
            updated_meta_df = updated_meta_df[updated_meta_df.index.isin(updated_uuids)]
        if updated_types is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_type'].isin(updated_types)]
        if updated_descriptions is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_description'].isin(updated_descriptions)]

        if updated_meas_df.empty:
            raise Exception("There are no database entries with the given uuids, types and descriptions.")

        updated_df = updated_benchmark.join_visualization_queries(updated_meas_df)

        updated_VisualizerClass = type_to_visualizer_class_mapper[updated_types[0]]
        updated_visualizer = updated_VisualizerClass(updated_df, 'plotly')
        updated_fig = updated_visualizer.plot_with_plotly()[0]

        updated_benchmark.close()

        return [dcc.Graph(id='fig_6', figure=updated_fig)]
    except:
        return ['No figure available']


# 7
@app.callback(
    Output('uuid_dropdown_7', 'options'),
    Input('db_file', 'value')
)
def filter_uuids_7(value):
    try:
        type_benchmark = VisualizationBenchmark(value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('type_dropdown_7', 'options'),
    Input('uuid_dropdown_7', 'value'),
    Input('db_file', 'value')
)
def filter_types_7(value, db_value):
    try:
        value_list = [value]
        type_benchmark = VisualizationBenchmark(db_value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        type_df = type_df[type_df['uuid'].isin(value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('desc_dropdown_7', 'options'),
    Input('uuid_dropdown_7', 'value'),
    Input('type_dropdown_7', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions_7(uuid_value, type_value, db_value):
    try:
        uuid_value_list = [uuid_value]
        type_value_list = [type_value]
        desc_benchmark = VisualizationBenchmark(db_value)
        desc_df = desc_benchmark.query_all_uuid_type_desc()
        desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
        desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
        desc_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('graph_7', 'children'),
    Input('uuid_dropdown_7', 'value'),
    Input('type_dropdown_7', 'value'),
    Input('desc_dropdown_7', 'value'),
    Input('db_file', 'value')
)
def update_graph_7(uuid_value, type_value, desc_value, db_value):
    try:
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        updated_benchmark = VisualizationBenchmark(db_value)
        updated_meas_df = updated_benchmark.query_all_uuid_type_desc()
        updated_meta_df = updated_benchmark.query_all_meta()

        if updated_uuids is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['uuid'].isin(updated_uuids)]
            updated_meta_df = updated_meta_df[updated_meta_df.index.isin(updated_uuids)]
        if updated_types is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_type'].isin(updated_types)]
        if updated_descriptions is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_description'].isin(updated_descriptions)]

        if updated_meas_df.empty:
            raise Exception("There are no database entries with the given uuids, types and descriptions.")

        updated_df = updated_benchmark.join_visualization_queries(updated_meas_df)

        updated_VisualizerClass = type_to_visualizer_class_mapper[updated_types[0]]
        updated_visualizer = updated_VisualizerClass(updated_df, 'plotly')
        updated_fig = updated_visualizer.plot_with_plotly()[0]

        updated_benchmark.close()

        return [dcc.Graph(id='fig_7', figure=updated_fig)]
    except:
        return ['No figure available']


# 8
@app.callback(
    Output('uuid_dropdown_8', 'options'),
    Input('db_file', 'value')
)
def filter_uuids_8(value):
    try:
        type_benchmark = VisualizationBenchmark(value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('type_dropdown_8', 'options'),
    Input('uuid_dropdown_8', 'value'),
    Input('db_file', 'value')
)
def filter_types_8(value, db_value):
    try:
        value_list = [value]
        type_benchmark = VisualizationBenchmark(db_value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        type_df = type_df[type_df['uuid'].isin(value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('desc_dropdown_8', 'options'),
    Input('uuid_dropdown_8', 'value'),
    Input('type_dropdown_8', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions_8(uuid_value, type_value, db_value):
    try:
        uuid_value_list = [uuid_value]
        type_value_list = [type_value]
        desc_benchmark = VisualizationBenchmark(db_value)
        desc_df = desc_benchmark.query_all_uuid_type_desc()
        desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
        desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
        desc_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('graph_8', 'children'),
    Input('uuid_dropdown_8', 'value'),
    Input('type_dropdown_8', 'value'),
    Input('desc_dropdown_8', 'value'),
    Input('db_file', 'value')
)
def update_graph_8(uuid_value, type_value, desc_value, db_value):
    try:
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        updated_benchmark = VisualizationBenchmark(db_value)
        updated_meas_df = updated_benchmark.query_all_uuid_type_desc()
        updated_meta_df = updated_benchmark.query_all_meta()

        if updated_uuids is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['uuid'].isin(updated_uuids)]
            updated_meta_df = updated_meta_df[updated_meta_df.index.isin(updated_uuids)]
        if updated_types is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_type'].isin(updated_types)]
        if updated_descriptions is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_description'].isin(updated_descriptions)]

        if updated_meas_df.empty:
            raise Exception("There are no database entries with the given uuids, types and descriptions.")

        updated_df = updated_benchmark.join_visualization_queries(updated_meas_df)

        updated_VisualizerClass = type_to_visualizer_class_mapper[updated_types[0]]
        updated_visualizer = updated_VisualizerClass(updated_df, 'plotly')
        updated_fig = updated_visualizer.plot_with_plotly()[0]

        updated_benchmark.close()

        return [dcc.Graph(id='fig_8', figure=updated_fig)]
    except:
        return ['No figure available']


# 9
@app.callback(
    Output('uuid_dropdown_9', 'options'),
    Input('db_file', 'value')
)
def filter_uuids_9(value):
    try:
        type_benchmark = VisualizationBenchmark(value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('type_dropdown_9', 'options'),
    Input('uuid_dropdown_9', 'value'),
    Input('db_file', 'value')
)
def filter_types_9(value, db_value):
    try:
        value_list = [value]
        type_benchmark = VisualizationBenchmark(db_value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        type_df = type_df[type_df['uuid'].isin(value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('desc_dropdown_9', 'options'),
    Input('uuid_dropdown_9', 'value'),
    Input('type_dropdown_9', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions_9(uuid_value, type_value, db_value):
    try:
        uuid_value_list = [uuid_value]
        type_value_list = [type_value]
        desc_benchmark = VisualizationBenchmark(db_value)
        desc_df = desc_benchmark.query_all_uuid_type_desc()
        desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
        desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
        desc_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('graph_9', 'children'),
    Input('uuid_dropdown_9', 'value'),
    Input('type_dropdown_9', 'value'),
    Input('desc_dropdown_9', 'value'),
    Input('db_file', 'value')
)
def update_graph_9(uuid_value, type_value, desc_value, db_value):
    try:
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        updated_benchmark = VisualizationBenchmark(db_value)
        updated_meas_df = updated_benchmark.query_all_uuid_type_desc()
        updated_meta_df = updated_benchmark.query_all_meta()

        if updated_uuids is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['uuid'].isin(updated_uuids)]
            updated_meta_df = updated_meta_df[updated_meta_df.index.isin(updated_uuids)]
        if updated_types is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_type'].isin(updated_types)]
        if updated_descriptions is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_description'].isin(updated_descriptions)]

        if updated_meas_df.empty:
            raise Exception("There are no database entries with the given uuids, types and descriptions.")

        updated_df = updated_benchmark.join_visualization_queries(updated_meas_df)

        updated_VisualizerClass = type_to_visualizer_class_mapper[updated_types[0]]
        updated_visualizer = updated_VisualizerClass(updated_df, 'plotly')
        updated_fig = updated_visualizer.plot_with_plotly()[0]

        updated_benchmark.close()

        return [dcc.Graph(id='fig_9', figure=updated_fig)]
    except:
        return ['No figure available']


# 10
@app.callback(
    Output('uuid_dropdown_10', 'options'),
    Input('db_file', 'value')
)
def filter_uuids_10(value):
    try:
        type_benchmark = VisualizationBenchmark(value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('type_dropdown_10', 'options'),
    Input('uuid_dropdown_10', 'value'),
    Input('db_file', 'value')
)
def filter_types_10(value, db_value):
    try:
        value_list = [value]
        type_benchmark = VisualizationBenchmark(db_value)
        type_df = type_benchmark.query_all_uuid_type_desc()
        type_df = type_df[type_df['uuid'].isin(value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
        type_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('desc_dropdown_10', 'options'),
    Input('uuid_dropdown_10', 'value'),
    Input('type_dropdown_10', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions_10(uuid_value, type_value, db_value):
    try:
        uuid_value_list = [uuid_value]
        type_value_list = [type_value]
        desc_benchmark = VisualizationBenchmark(db_value)
        desc_df = desc_benchmark.query_all_uuid_type_desc()
        desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
        desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
        choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
        desc_benchmark.close()
        return choices
    except:
        return []


@app.callback(
    Output('graph_10', 'children'),
    Input('uuid_dropdown_10', 'value'),
    Input('type_dropdown_10', 'value'),
    Input('desc_dropdown_10', 'value'),
    Input('db_file', 'value')
)
def update_graph_10(uuid_value, type_value, desc_value, db_value):
    try:
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        updated_benchmark = VisualizationBenchmark(db_value)
        updated_meas_df = updated_benchmark.query_all_uuid_type_desc()
        updated_meta_df = updated_benchmark.query_all_meta()

        if updated_uuids is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['uuid'].isin(updated_uuids)]
            updated_meta_df = updated_meta_df[updated_meta_df.index.isin(updated_uuids)]
        if updated_types is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_type'].isin(updated_types)]
        if updated_descriptions is not None:
            updated_meas_df = updated_meas_df[updated_meas_df['measurement_description'].isin(updated_descriptions)]

        if updated_meas_df.empty:
            raise Exception("There are no database entries with the given uuids, types and descriptions.")

        updated_df = updated_benchmark.join_visualization_queries(updated_meas_df)

        updated_VisualizerClass = type_to_visualizer_class_mapper[updated_types[0]]
        updated_visualizer = updated_VisualizerClass(updated_df, 'plotly')
        updated_fig = updated_visualizer.plot_with_plotly()[0]

        updated_benchmark.close()

        return [dcc.Graph(id='fig_10', figure=updated_fig)]
    except:
        return ['No figure available']


if __name__ == '__main__':
    app.run_server(debug=True)
