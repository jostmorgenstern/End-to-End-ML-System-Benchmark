import time

import numpy as np

from e2ebench import Benchmark,\
                     ConfusionMatrixTracker,\
                     HyperparameterTracker,\
                     BenchmarkSupervisor,\
                     TimeMetric,\
                     MemoryMetric,\
                     ThroughputMetric,\
                     LatencyMetric,\
                     TTATracker,\
                     LossTracker,\
                     PowerMetric,\
                     EnergyMetric
                     LossTracker,\
                     CPUMetric
 

bm = Benchmark('sample_db_file.db', description="le description")

bloat_metrics = {
    "throughput": ThroughputMetric('bloat throughput'),
    "latency": LatencyMetric('bloat latency'),
    "time": TimeMetric('bloat time'),
    "memory": MemoryMetric('bloat memory', interval=0.1),
    "power": PowerMetric('bloat power'),
    "energy": EnergyMetric('bloat energy'),
    "cpu": CPUMetric('bloat cpu', interval=0.1)
}


@BenchmarkSupervisor(bloat_metrics.values(), bm)
def bloat():
    a = []
    for i in range(1, 2):
        a.append(np.random.randn(*([10] * i)))
        time.sleep(5)
    print(a)
    bloat_metrics["throughput"].track(420)
    bloat_metrics["latency"].track(69)


def main():
    conf_mat = np.arange(9).reshape((3, 3))
    labels = ['foo', 'bar', 'baz']
    ConfusionMatrixTracker(bm).track(conf_mat, labels, 'foobar')

    with HyperparameterTracker(bm, "hyper params of sample pipeline", ['lr', 'num_epochs', 'num_layers'], 'loss') as ht:
        ht.track({'lr': 0.03, 'num_epochs': 10, 'num_layers': 4, 'loss': 42})
        ht.track({'lr': 0.08, 'num_epochs': 15, 'num_layers': 2, 'loss': 69})

    bloat()

    losses = np.random.randint(0, 100, size=100)
    LossTracker(bm).track(losses, "loss values for training run 42 of sample pipeline")
    TTAs = np.random.randint(0, 100, size=100)
    TTATracker(bm).track(TTAs, "TTA values for training run 42 of sample pipeline")

    bm.close()

    a = 0


if __name__ == "__main__":
    main()
