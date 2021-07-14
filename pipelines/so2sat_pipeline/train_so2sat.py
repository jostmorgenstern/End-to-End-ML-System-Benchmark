import h5py
import e2ebench as eb
from benchmarking import bm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


class DatasetGenerator:
    def __init__(self, file):
        self.file = file

    def __call__(self, *args, **kwargs):
        with h5py.File(self.file,  'r') as f:
            # for i in range(len(f['label'])):
            for i in range(4096):
                yield f['sen1'][i], f['label'][i]



# def load_data(num_samples=None):
#     f = h5py.File('data/training.h5', 'r')
#     n = num_samples or len(f['label'])  # if num_samples is 0 or None, use all samples
#     input_train = f['sen1'][0:n]
#     label_train = f['label'][0:n]
#     f.close()
#     f = h5py.File('data/validation.h5', 'r')
#     input_val = f['sen1'][0:len(f['label'])]
#     label_val = f['label'][0:len(f['label'])]
#     f.close()
#     return input_train, label_train, input_val, label_val, n


def load_data():
    train_ds = tf.data.Dataset.from_generator(DatasetGenerator('data/training.h5'),
                                              output_signature=(tf.TensorSpec(shape=(32, 32, 8),
                                                                              dtype=tf.float64),
                                                                tf.TensorSpec(shape=17,
                                                                              dtype=tf.float64)))

    validation_ds = tf.data.Dataset.from_generator(DatasetGenerator('data/validation.h5'),
                                                   output_signature=(tf.TensorSpec(shape=(32, 32, 8),
                                                                                   dtype=tf.float64),
                                                                     tf.TensorSpec(shape=17,
                                                                                   dtype=tf.float64)))

    return train_ds, validation_ds


def compile_model(input_shape, num_classes, loss_function, optimizer):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])

    model.summary()

    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


throughput_metric = eb.ThroughputMetric('train throughput')
latency_metric = eb.LatencyMetric('train latency')


@eb.BenchmarkSupervisor([eb.MemoryMetric('train memory'),
                         eb.TimeMetric('train time'),
                         eb.CPUMetric('train cpu usage')
                         # throughput_metric,
                         # latency_metric
                         ], bm)
def train():
    per_worker_batch_size = 128
    input_shape = (32, 32, 8)
    loss_function = "categorical_crossentropy"
    num_classes = 17
    num_epochs = 10
    optimizer = Adam()
    verbosity = 1

    # input_train, label_train, input_val, label_val, num_samples = load_data()
    #
    # input_train = input_train.reshape((len(input_train), img_width, img_height, img_num_channels))
    # input_val = input_val.reshape((len(input_val), img_width, img_height, img_num_channels))

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        global_batch_size = per_worker_batch_size * strategy.num_replicas_in_sync

        train_ds, val_ds = load_data()
        # dist_val_ds = strategy.experimental_distribute_dataset(val_ds)

        model = compile_model(input_shape, num_classes, loss_function, optimizer)

        history = model.fit(train_ds.batch(global_batch_size),
                            epochs=num_epochs,
                            verbose=verbosity,
                            validation_data=val_ds)

    # throughput_metric.track(num_samples / num_workers)
    # latency_metric.track(num_samples / num_workers)

    return {"model": model, "classifier": optimizer,
            "accuracy": history.history["accuracy"]}
