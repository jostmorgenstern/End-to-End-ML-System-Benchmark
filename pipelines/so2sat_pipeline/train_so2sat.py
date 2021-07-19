import h5py
import e2ebench as eb
from benchmarking import bm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


class DatasetGenerator:
    def __init__(self, file, num_samples=None):
        self.file = file
        self.num_samples = num_samples

    def __call__(self, *args, **kwargs):
        with h5py.File(self.file, 'r') as f:
            n = self.num_samples or len(f['label'])
            for i in range(n):
                yield (np.concatenate((f['sen1'][i], f['sen2'][i]), axis=2),
                       (f['label'][i]))


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


@eb.BenchmarkSupervisor([eb.MemoryMetric('train memory', interval=4),
                         eb.TimeMetric('train time'),
                         eb.CPUMetric('train cpu usage', interval=4),
                         throughput_metric,
                         latency_metric
                         ], bm)
def scope_func(strategy, per_worker_batch_size, num_epochs,
               input_shape, num_classes, loss_function, optimizer, verbosity):
    num_workers = strategy.num_replicas_in_sync

    global_batch_size = per_worker_batch_size * num_workers

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_ds = tf.data.Dataset.from_generator(DatasetGenerator('data/training.h5'),
                                              output_signature=(tf.TensorSpec(shape=(32, 32, 18),
                                                                              dtype=tf.float64),
                                                                tf.TensorSpec(shape=17,
                                                                              dtype=tf.float64)))

    train_ds = train_ds.with_options(options).batch(global_batch_size)
    val_ds = tf.data.Dataset.from_generator(DatasetGenerator('data/validation.h5'),
                                            output_signature=(tf.TensorSpec(shape=(32, 32, 18),
                                                                            dtype=tf.float64),
                                                              tf.TensorSpec(shape=17,
                                                                            dtype=tf.float64)))
    val_ds = val_ds.with_options(options).batch(global_batch_size)

    train_ds.prefetch(10)
    val_ds.prefetch(10)

    model = compile_model(input_shape, num_classes, loss_function, optimizer)

    history = model.fit(train_ds,
                        epochs=num_epochs,
                        verbose=verbosity,
                        validation_data=val_ds)

    with h5py.File('data/training.h5', 'r') as f:
        num_samples = len(f['label'])

    throughput_metric.track((num_samples / num_workers) * num_epochs)
    latency_metric.track((num_samples / num_workers) * num_epochs)

    return model, history


def train():
    per_worker_batch_size = 512
    input_shape = (32, 32, 18)
    loss_function = "categorical_crossentropy"
    num_classes = 17
    num_epochs = 5
    optimizer = Adam(learning_rate=0.0005)
    verbosity = 1

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        model, history = scope_func(strategy, per_worker_batch_size, num_epochs,
                                    input_shape, num_classes, loss_function, optimizer, verbosity)

    return {"model": model, "classifier": optimizer,
            "accuracy": history.history["accuracy"]}
