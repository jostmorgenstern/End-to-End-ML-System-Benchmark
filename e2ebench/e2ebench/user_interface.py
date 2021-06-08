import base64
import io
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import flask
from dash.dependencies import Input, Output
from e2ebench import VisualizationBenchmark
from e2ebench.visualization import type_to_visualizer_class_mapper
from plotly.subplots import make_subplots

external_stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheet)
server = app.server
#
# example_db = '/Users/christiancarljacob/PycharmProjects/DESEnd-to-end-ML-System-Benchmark/pipelines/sample_pipeline/sample_db_file.db'
# stock_db = '/Users/christiancarljacob/PycharmProjects/DESEnd-to-end-ML-System-Benchmark/pipelines/stock_market_pipeline/stock_market_benchmark.db'
#
# db_file = example_db
#
# example_uuids = ['10101768-963a-4104-a770-35ae2a04cb92']
# example_types = ['confusion-matrix']
# example_descriptions = ['foobar']
#
# stock_uuids = ['ff82a9ee-5822-41ea-a058-a712656cb039']
# stock_types = ['memory']
# stock_descriptions = ['prep memory']
#
# uuids = example_uuids
# types = example_types
# descriptions = example_descriptions


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


# def filter_by_args(meas_df, meta_df):
#     if uuids is not None:
#         meas_df = meas_df[meas_df['uuid'].isin(uuids)]
#         meta_df = meta_df[meta_df.index.isin(uuids)]
#     if types is not None:
#         meas_df = meas_df[meas_df['measurement_type'].isin(types)]
#     if descriptions is not None:
#         meas_df = meas_df[meas_df['measurement_description'].isin(descriptions)]
#
#     if meas_df.empty:
#         raise Exception("There are no database entries with the given uuids, types and descriptions.")
#
#     return meas_df, meta_df


# benchmark = VisualizationBenchmark(db_file)
# meas_df = benchmark.query_all_uuid_type_desc()
# meta_df = benchmark.query_all_meta()
# meas_df, meta_df = filter_by_args(meas_df, meta_df)
#
# df = benchmark.join_visualization_queries(meas_df)
#
# VisualizerClass = type_to_visualizer_class_mapper[types[0]]
# visualizer = VisualizerClass(df, 'plotly')
# fig = visualizer.plot_with_plotly()[0]

# benchmark.close()

app.layout = html.Div(
    children=[
        html.H1(children="e2ebench"),
        html.H3(children="Sample pipeline"),

        dcc.Upload(
            id='db_upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            }
        ),
        html.Div(id='test_upload'),


        dcc.Input(id='db_file', type='text', placeholder='Enter the path to your db-file here',
                  value='/Users/christiancarljacob/PycharmProjects/DESEnd-to-end-ML-System-Benchmark/pipelines/sample_pipeline/sample_db_file.db',
                  debounce=True),

        html.Div(id='db_values'),

        html.Div([
            html.Label('UUID'),
            dcc.Dropdown(
                id='uuid_dropdown',
                # options=[{'label': x, 'value': x} for x in get_all_uuids(benchmark.query_all_uuid_type_desc())],
                # value='10101768-963a-4104-a770-35ae2a04cb92'
            ),
        ]),
        html.Div([
            html.Label('Type'),
            dcc.Dropdown(
                id='type_dropdown',
                # options=[{'label': x, 'value': x} for x in get_all_types(benchmark.query_all_uuid_type_desc())],
                # value='confusion-matrix'
            ),
        ]),
        html.Div([
            html.Label('Description'),
            dcc.Dropdown(
                id='desc_dropdown',
                # options=[{'label': x, 'value': x} for x in get_all_descriptions(benchmark.query_all_uuid_type_desc())],
                # value='foobar'
            ),
        ]),
        html.Br(),
        html.Div(id='graph_1'),
        # html.Br(),
        # html.Button('Show all', id='show_all_1', n_clicks=0),
        # html.Div(id='all_graphs_1', hidden=True),
        html.Br(),
        html.Button('Show/ Hide second Section', id='add_button', n_clicks=0),
        html.Div(
            id='section_2',
            children=[
                html.H3('Second Section'),
                html.Div([
                    html.Label('UUID'),
                    dcc.Dropdown(
                        id='uuid_dropdown_2',
                        # options=[{'label': x, 'value': x} for x in get_all_uuids(benchmark.query_all_uuid_type_desc())],
                        # value='10101768-963a-4104-a770-35ae2a04cb92'
                    ),
                ]),
                html.Div([
                    html.Label('Type'),
                    dcc.Dropdown(
                        id='type_dropdown_2',
                        # options=[{'label': x, 'value': x} for x in get_all_types(benchmark.query_all_uuid_type_desc())],
                        # value='confusion-matrix'
                    ),
                ]),
                html.Div([
                    html.Label('Description'),
                    dcc.Dropdown(
                        id='desc_dropdown_2',
                        # options=[{'label': x, 'value': x} for x in
                                 # get_all_descriptions(benchmark.query_all_uuid_type_desc())],
                        # value='foobar'
                    ),
                ]),
                html.Br(),
                html.Div(id='graph_2'),
                html.Br(),
            ],
            hidden=False)
    ])


# @app.callback(
#     Output('all_graphs_1', 'children'),
#     Input('uuid_dropdown', 'value'),
#     Input('type_dropdown', 'value'),
#     Input('desc_dropdown', 'value'),
#     Input('db_file', 'value')
# )
# def show_all_graphs_1(uuid_value, type_value, desc_value, db_file_value):
#     subplot_fig = make_subplots(
#         rows=1, cols=2,
#     )
#     updated_uuids = [uuid_value]
#     updated_types = [type_value]
#     updated_descriptions = [desc_value]
#
#     updated_benchmark = VisualizationBenchmark(db_file_value)
#     updated_meas_df = updated_benchmark.query_all_uuid_type_desc()
#     updated_meta_df = updated_benchmark.query_all_meta()
#
#     if updated_uuids is not None:
#         updated_meas_df = updated_meas_df[updated_meas_df['uuid'].isin(updated_uuids)]
#         updated_meta_df = updated_meta_df[updated_meta_df.index.isin(updated_uuids)]
#     if updated_types is not None:
#         updated_meas_df = updated_meas_df[updated_meas_df['measurement_type'].isin(updated_types)]
#     if updated_descriptions is not None:
#         updated_meas_df = updated_meas_df[updated_meas_df['measurement_description'].isin(updated_descriptions)]
#
#     if updated_meas_df.empty:
#         raise Exception("There are no database entries with the given uuids, types and descriptions.")
#
#     updated_df = updated_benchmark.join_visualization_queries(updated_meas_df)
#
#     updated_VisualizerClass = type_to_visualizer_class_mapper[updated_types[0]]
#     updated_visualizer = updated_VisualizerClass(updated_df, 'plotly')
#     updated_fig = updated_visualizer.plot_with_plotly()[0]
#
#
#
#     updated_benchmark.close()



@app.callback(
    Output('test_upload', 'children'),
    Input('db_upload', 'contents')
)
def update_db_filename(filename):
    return filename


@app.callback(
    Output('db_values', 'children'),
    Input('db_upload', 'contents')
)
def store_db_values(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    file_like_object = io.BytesIO(decoded)
    return file_like_object


@app.callback(
    Output('section_2', 'hidden'),
    Input('add_button', 'n_clicks'),
    Input('section_2', 'hidden')
)
def enable_sec_2(clicks, current_state):
    if current_state:
        return False
    else:
        return True


@app.callback(
    Output('uuid_dropdown', 'options'),
    Input('db_file', 'value')
)
def filter_uuids(value):
    type_benchmark = VisualizationBenchmark(value)
    type_df = type_benchmark.query_all_uuid_type_desc()
    choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
    type_benchmark.close()
    return choices


@app.callback(
    Output('type_dropdown', 'options'),
    Input('uuid_dropdown', 'value'),
    Input('db_file', 'value')
)
def filter_types(value, db_value):
    value_list = [value]
    type_benchmark = VisualizationBenchmark(db_value)
    type_df = type_benchmark.query_all_uuid_type_desc()
    type_df = type_df[type_df['uuid'].isin(value_list)]
    choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
    type_benchmark.close()
    return choices


@app.callback(
    Output('desc_dropdown', 'options'),
    Input('uuid_dropdown', 'value'),
    Input('type_dropdown', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions(uuid_value, type_value, db_value):
    uuid_value_list = [uuid_value]
    type_value_list = [type_value]
    desc_benchmark = VisualizationBenchmark(db_value)
    desc_df = desc_benchmark.query_all_uuid_type_desc()
    desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
    desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
    choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
    desc_benchmark.close()
    return choices


@app.callback(
    Output('uuid_dropdown_2', 'options'),
    Input('db_file', 'value')
)
def filter_uuids_2(value):
    type_benchmark = VisualizationBenchmark(value)
    type_df = type_benchmark.query_all_uuid_type_desc()
    choices = [{'label': x, 'value': x} for x in get_all_uuids(type_df)]
    type_benchmark.close()
    return choices


@app.callback(
    Output('type_dropdown_2', 'options'),
    Input('uuid_dropdown_2', 'value'),
    Input('db_file', 'value')
)
def filter_types_2(value, db_value):
    value_list = [value]
    type_benchmark = VisualizationBenchmark(db_value)
    type_df = type_benchmark.query_all_uuid_type_desc()
    type_df = type_df[type_df['uuid'].isin(value_list)]
    choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
    type_benchmark.close()
    return choices


@app.callback(
    Output('desc_dropdown_2', 'options'),
    Input('uuid_dropdown_2', 'value'),
    Input('type_dropdown_2', 'value'),
    Input('db_file', 'value')
)
def filter_descriptions_2(uuid_value, type_value, db_value):
    uuid_value_list = [uuid_value]
    type_value_list = [type_value]
    desc_benchmark = VisualizationBenchmark(db_value)
    desc_df = desc_benchmark.query_all_uuid_type_desc()
    desc_df = desc_df[desc_df['uuid'].isin(uuid_value_list)]
    desc_df = desc_df[desc_df['measurement_type'].isin(type_value_list)]
    choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
    desc_benchmark.close()
    return choices


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
        return []


@app.callback(
    Output('graph_2', 'children'),
    Input('uuid_dropdown_2', 'value'),
    Input('type_dropdown_2', 'value'),
    Input('desc_dropdown_2', 'value'),
    Input('test_upload', 'children'),
    Input('db_upload', 'contents'),
    Input('db_upload', 'filename')
)
def update_graph_2(uuid_value, type_value, desc_value, db_value, content, filename):
    try:
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        file_like_object = io.BytesIO(decoded)

        path = f'db_files/{filename}'

        updated_benchmark = VisualizationBenchmark(path)
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


@app.server.route('/db_files/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(
        os.path.join(root_dir, 'db_files'), path
    )


if __name__ == '__main__':
    app.run_server(debug=True)
