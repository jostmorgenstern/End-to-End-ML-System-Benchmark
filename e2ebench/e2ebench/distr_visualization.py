import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from collections import namedtuple, defaultdict

cmaps = [
    cm.get_cmap(name='spring'),
    cm.get_cmap(name='summer'),
    cm.get_cmap(name='cool'),
    cm.get_cmap(name='Wistia'),
    cm.get_cmap(name='winter'),
    cm.get_cmap(name='autumn')
]


def pyplot_color_to_plotly_color(color):
    pyplot_color = []
    for i in range(3):
        pyplot_color.append(round(color[i] * 255))
    pyplot_color.append(color[3])
    return f'rgba{tuple(pyplot_color)}'


class Visualizer:
    title = None
    xaxis_label = None
    yaxis_label = None

    def __init__(self, df_from_cli, plotting_backend):
        self.df = df_from_cli
        self.plotting_backend = plotting_backend
        self.color_maps = iter(cmaps)

    def plot(self):
        if self.plotting_backend == 'matplotlib':
            return self.plot_with_matplotlib()
        if self.plotting_backend == 'plotly':
            return self.plot_with_plotly()

    def plot_with_matplotlib(self):
        pass

    def plot_with_plotly(self):
        pass

    def get_next_color_map(self):
        try:
            color_map = next(self.color_maps)
        except StopIteration:
            self.color_maps = iter(cmaps)
            color_map = next(self.color_maps)
        return color_map


class WorkerAggregateBarVisualizer(Visualizer):
    def __init__(self, df_from_cli, plotting_backend):
        super().__init__(df_from_cli, plotting_backend)

        self.df = self.df.sort_values(by='measurement_datetime', ascending=False)

    def plot_with_matplotlib(self):
        figs = []

        for measurement_description, df in self.df.groupby('measurement_description'):
            fig, ax = plt.subplots()
            sns.histplot(data=df,
                         x='meta_description',
                         weights='measurement_data',
                         hue='worker_number',
                         multiple='stack',
                         ax=ax)

            print()

            ax.tick_params(axis='x', labelrotation=75)
            fig.tight_layout()

            ax.set_xlabel(self.xaxis_label)
            ax.set_ylabel(self.yaxis_label)
            ax.set_title(measurement_description)

            figs.append(fig)
        return figs

    def plot_with_plotly(self):
        pass


class WorkerMaximumBarVisualizer(Visualizer):
    def __init__(self, df_from_cli, plotting_backend):
        super().__init__(df_from_cli, plotting_backend)

        self.data_df = self.df.groupby(['uuid', 'measurement_description']).max().reset_index()
        self.data_df = self.data_df.sort_values(by='measurement_datetime', ascending=False)

    def plot_with_matplotlib(self):
        figs = []

        for measurement_description, df in self.data_df.groupby('measurement_description'):
            fig, ax = plt.subplots()
            sns.barplot(x='meta_description',
                        y='measurement_data',
                        data=df,
                        ax=ax)
            ax.tick_params(axis='x', labelrotation=75)
            fig.tight_layout()

            ax.set_xlabel(self.xaxis_label)
            ax.set_ylabel(self.yaxis_label)
            ax.set_title(measurement_description)

            figs.append(fig)
        return figs

    def plot_with_plotly(self):
        pass


class TimebasedMultiLineChartVisualizer(Visualizer):
    def __init__(self, df_from_cli, plotting_backend):
        super().__init__(df_from_cli, plotting_backend)
        self.df = self.df.sort_values(by='measurement_datetime', ascending=False)

    def plot_with_matplotlib(self):
        figs = []

        def flip_dict(d):
            return [dict(zip(d, i)) for i in zip(*d.values())]

        for meta_description, meta_df in self.df.groupby('meta_description'):
            fig, ax = plt.subplots()

            meta_df['measurement_data'] = meta_df['measurement_data'].apply(flip_dict)

            meta_df = meta_df.explode('measurement_data', ignore_index=True)
            meta_df[['timestamp', 'value']] = meta_df['measurement_data'].apply(pd.Series)
            meta_df['timedelta'] = (meta_df['timestamp'] - meta_df['meta_start_time'])
            meta_df['timedelta'] = meta_df['timedelta'].apply(lambda x: x.total_seconds())

            sns.lineplot(data=meta_df,
                         x='timedelta',
                         y='value',
                         style='measurement_description',
                         hue='worker_number',
                         ax=ax)

            ax.set_xlabel(self.xaxis_label)
            ax.set_ylabel(self.yaxis_label)
            ax.set_title(meta_description)

            figs.append(fig)

        return figs

    def plot_with_plotly(self):
        pass


# class TimebasedMultiLineChartVisualizer(Visualizer):
#     def __init__(self, df_from_cli, plotting_backend):
#         super().__init__(df_from_cli, plotting_backend)
#
#         self.data_dict = defaultdict(list)
#         LineData = namedtuple('LineData', ('label', 'measurements', 'times', 'worker_number'))
#         LineMeta = namedtuple('LineMeta', ('uuid', 'description', 'start_time'))
#
#         for uuid, df in self.df.groupby('uuid'):
#             for row in df.itertuples():
#                 start_time = pd.to_datetime(row.meta_start_time)
#                 timestamps = pd.to_datetime(row.measurement_data['timestamps'])
#                 times = [timedelta.total_seconds() for timedelta in (timestamps - start_time)]
#
#                 label = f'"{row.measurement_description}" of {row.meta_description}'
#                 if row.worker_number is not None:
#                     label += f' (worker {row.worker_number})'
#
#                 line_data = LineData(label=label,
#                                      measurements=row.measurement_data['measurements'],
#                                      times=times,
#                                      worker_number=row.worker_number)
#                 line_meta = LineMeta(uuid=uuid,
#                                      description=row.measurement_description,
#                                      start_time=row.meta_start_time)
#                 self.data_dict[line_meta].append(line_data)
#
#     # def plot_with_plotly(self):
#     #
#     #     fig = go.Figure()
#     #
#     #     for group in self.data_dict:
#     #         color_map = self.get_next_color_map()
#     #         group_size = len(self.data_dict[group])
#     #         for i in range(group_size):
#     #             line_data = self.data_dict[group][i]
#     #             color = color_map(i / group_size)
#     #             fig.add_trace(go.Scatter(
#     #                 x=line_data.times,
#     #                 y=line_data.measurements,
#     #                 mode='lines+markers',
#     #                 name=line_data.label,
#     #                 line={'color': pyplot_color_to_plotly_color(color)}
#     #             ))
#     #
#     #
#     #     fig.update_layout(
#     #         title=self.title,
#     #         xaxis_title=self.xaxis_label,
#     #         yaxis_title=self.yaxis_label,
#     #     )
#     #
#     #     return [fig]
#
#     def plot_with_matplotlib(self):
#         figs = []
#         for group in self.data_dict:
#             fig, ax = plt.subplots()
#             for line_data in self.data_dict[group]:
#                 ax.plot(line_data.times, line_data.measurements, label=line_data.label)
#             ax.set_xlabel(self.xaxis_label)
#             ax.set_ylabel(self.yaxis_label)
#             ax.set_title(self.title)
#             ax.set_ylim(bottom=0)
#             ax.legend()
#
#             figs.append(fig)
#
#         return figs
#
#     # def plot_with_matplotlib(self):
#     #     fig, ax = plt.subplots()
#     #
#     #     for group in self.data_dict:
#     #         color_map = self.get_next_color_map()
#     #         group_size = len(self.data_dict[group])
#     #         for i in range(group_size):
#     #             line_data = self.data_dict[group][i]
#     #             color = color_map(i/group_size)
#     #             ax.plot(line_data.times, line_data.measurements, label=line_data.label, c=color)
#     #
#     #     ax.set_xlabel(self.xaxis_label)
#     #     ax.set_ylabel(self.yaxis_label)
#     #     ax.set_title(self.title)
#     #
#
#     #     ax.yaxis.set_major_locator(ticker.LinearLocator(12))
#     #
#     #     return [fig]


class MemoryVisualizer(TimebasedMultiLineChartVisualizer):
    title = "Metric: Memory usage"
    xaxis_label = "Seconds elapsed since start of pipeline run"
    yaxis_label = "Memory usage in MB"


class CPUVisualizer(TimebasedMultiLineChartVisualizer):
    title = "Metric: CPU usage"
    xaxis_label = "Seconds elapsed since start of pipeline run"
    yaxis_label = "CPU usage in %"


class TimeVisualizer(WorkerMaximumBarVisualizer):
    title = "Metric: Time"
    xaxis_label = ""
    yaxis_label = "Time taken in seconds"


class ThroughputVisualizer(WorkerAggregateBarVisualizer):
    title = "Metric: Throughput"
    xaxis_label = ""
    yaxis_label = "Throughput in entries per second"


class LatencyVisualizer(WorkerAggregateBarVisualizer):
    title = "Metric: Latency"
    xaxis_label = ""
    yaxis_label = "Latency in Seconds/entry"


type_to_visualizer_class_mapper = {
    "memory": MemoryVisualizer,
    "cpu": CPUVisualizer,
    "time": TimeVisualizer,
    "throughput": ThroughputVisualizer,
    "latency": LatencyVisualizer
}
