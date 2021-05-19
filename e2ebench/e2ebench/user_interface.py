import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from e2ebench import VisualizationBenchmark
from e2ebench.visualization import type_to_visualizer_class_mapper

external_stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheet)

db_file = '/Users/christiancarljacob/PycharmProjects/DESEnd-to-end-ML-System-Benchmark/pipelines/sample_pipeline/sample_db_file.db'

uuids = ['10101768-963a-4104-a770-35ae2a04cb92']
types = ['throughput']
descriptions = ['bloat throughput']


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


def filter_by_args(meas_df, meta_df):
    if uuids is not None:
        meas_df = meas_df[meas_df['uuid'].isin(uuids)]
        meta_df = meta_df[meta_df.index.isin(uuids)]
    if types is not None:
        meas_df = meas_df[meas_df['measurement_type'].isin(types)]
    if descriptions is not None:
        meas_df = meas_df[meas_df['measurement_description'].isin(descriptions)]

    if meas_df.empty:
        raise Exception("There are no database entries with the given uuids, types and descriptions.")

    return meas_df, meta_df


benchmark = VisualizationBenchmark(db_file)
meas_df = benchmark.query_all_uuid_type_desc()
meta_df = benchmark.query_all_meta()
meas_df, meta_df = filter_by_args(meas_df, meta_df)

df = benchmark.join_visualization_queries(meas_df)

VisualizerClass = type_to_visualizer_class_mapper['throughput']
visualizer = VisualizerClass(df, 'plotly')
fig = visualizer.plot_with_plotly()[0]

benchmark.close()

app.layout = html.Div(children=[
    html.H1(children="e2ebench"),
    html.H3(children="Sample pipeline"),

    html.Div([
        html.Label('UUID'),
        dcc.Dropdown(
            id='uuid_dropdown',
            options=[{'label': x, 'value': x} for x in get_all_uuids(benchmark.query_all_uuid_type_desc())]#,
            # value='10101768-963a-4104-a770-35ae2a04cb92'
        ),
    ]),
    html.Div([
        html.Label('Type'),
        dcc.Dropdown(
            id='type_dropdown',
            options=[{'label': x, 'value': x} for x in get_all_types(benchmark.query_all_uuid_type_desc())]#,
            # value='throughput'
        ),
    ]),
    html.Div([
        html.Label('Description'),
        dcc.Dropdown(
            id='desc_dropdown',
            options=[{'label': x, 'value': x} for x in get_all_descriptions(benchmark.query_all_uuid_type_desc())]#,
            # value='bloat throughput'
        ),
    ]),
    html.Br(),
    html.Div([
        dcc.Graph(
            id="test_graph",
            figure=fig
        )
    ])
])


@app.callback(
    Output('type_dropdown', 'options'),
    Input('uuid_dropdown', 'value')
)
def filter_types(value):
    value_list = [value]
    type_benchmark = VisualizationBenchmark(db_file)
    type_df = type_benchmark.query_all_uuid_type_desc()
    type_df = type_df[type_df['uuid'].isin(value_list)]
    choices = [{'label': x, 'value': x} for x in get_all_types(type_df)]
    type_benchmark.close()
    return choices


@app.callback(
    Output('desc_dropdown', 'options'),
    Input('type_dropdown', 'value')
)
def filter_descriptions(value):
    value_list = [value]
    desc_benchmark = VisualizationBenchmark(db_file)
    desc_df = desc_benchmark.query_all_uuid_type_desc()
    desc_df = desc_df[desc_df['measurement_type'].isin(value_list)]
    choices = [{'label': x, 'value': x} for x in get_all_descriptions(desc_df)]
    desc_benchmark.close()
    return choices


@app.callback(
    Output('test_graph', 'figure'),
    Input('uuid_dropdown', 'value'),
    Input('type_dropdown', 'value'),
    Input('desc_dropdown', 'value'),
    Input('test_graph', 'figure')
)
def update_chart_uuid(uuid_value, type_value, desc_value, old_figure):
    if uuid_value != '' and type_value != '' and desc_value != '':
        updated_uuids = [uuid_value]
        updated_types = [type_value]
        updated_descriptions = [desc_value]

        updated_benchmark = VisualizationBenchmark(db_file)
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

        return updated_fig
    else:
        return old_figure


if __name__ == '__main__':
    app.run_server(debug=True)
