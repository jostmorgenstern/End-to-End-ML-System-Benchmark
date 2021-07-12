import h5py
from sklearn.metrics import confusion_matrix
import numpy as np
from benchmarking import bm
import e2ebench as eb


def load_data(num_samples=0):
    f = h5py.File('data/testing.h5', 'r')
    n = num_samples or len(f['label'])  # if num_samples is 0 or None, use all samples
    input_test = f['sen1'][0:n]
    label_test = f['label'][0:n]
    f.close()

    return input_test, label_test, num_samples


latency_metric = eb.LatencyMetric('test latency')
throughput_metric = eb.ThroughputMetric('test throughput')


@eb.BenchmarkSupervisor([eb.MemoryMetric('test memory'),
                         eb.TimeMetric('test time'),
                         eb.CPUMetric('test cpu usage'),
                         latency_metric,
                         throughput_metric
                         ], bm)
def test(model):

    img_width, img_height, img_num_channels = 32, 32, 8

    input_test, label_test, num_samples = load_data()

    input_test = input_test.reshape((len(input_test), img_width, img_height, img_num_channels))

    # Generate generalization metrics
    score = model.evaluate(input_test, label_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    # Generate confusion matrix
    pred_test = model.predict_classes(input_test)
    label_test = np.argmax(label_test, axis=1)

    con_mat = confusion_matrix(label_test, pred_test)

    classes = ["compact high-rise", "compact mid-rise", "compact low-rise",
               "open high-rise", "open mid-rise", "open low-rise",
               "lightweight low-rise", "large low-rise", "sparsely built",
               "heavy industry", "dense trees", "scattered tree",
               "brush, scrub", "low plants", "bare rock or paved",
               "bare soil or sand", "water"]

    latency_metric.track(num_samples)
    throughput_metric.track(num_samples)

    return {"confusion matrix": con_mat, "classes": classes}
