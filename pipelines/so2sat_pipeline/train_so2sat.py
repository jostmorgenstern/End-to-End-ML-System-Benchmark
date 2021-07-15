import h5py
import e2ebench as eb
from benchmarking import bm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


class DatasetGenerator:
    def __init__(self, file, global_batch_size):
        self.file = file
        self.global_batch_size = global_batch_size

    def __call__(self, *args, **kwargs):
        with h5py.File(self.file, 'r') as f:
            file_len = len(f['label'])
            start_index = 0
            end_index = self.global_batch_size - 1
            while start_index < file_len:
                yield (np.concatenate((f['sen1'][start_index:end_index], f['sen2'][start_index:end_index]), axis=3),
                       f['label'][start_index:end_index])
                start_index += self.global_batch_size
                end_index += self.global_batch_size


def load_data(num_samples=None):
    with h5py.File('data/training.h5', 'r') as train_f:
        n = num_samples or len(train_f['label'])  # if num_samples is 0 or None, use all samples
        input_train = np.concatenate((train_f['sen1'][0:n], train_f['sen2'][0:n]), axis=3)
        label_train = train_f['label'][0:n]
    with h5py.File('data/validation.h5', 'r') as val_f:
        input_val = np.concatenate((val_f['sen1'][0:n], val_f['sen2'][0:n]), axis=3)
        label_val = val_f['label'][0:len(val_f['label'])]
    return input_train, label_train, input_val, label_val, n


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
                         eb.CPUMetric('train cpu usage'),
                         throughput_metric,
                         latency_metric
                         ], bm)
def train():
    per_worker_batch_size = 64
    input_shape = (32, 32, 18)
    loss_function = "categorical_crossentropy"
    num_classes = 17
    num_epochs = 10
    optimizer = Adam()
    verbosity = 1

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        num_workers = strategy.num_replicas_in_sync

        global_batch_size = per_worker_batch_size * num_workers

        # input_train, label_train, input_val, label_val, num_samples = load_data(num_samples=32768)
        # train_ds = tf.data.Dataset.from_tensor_slices((input_train, label_train)).batch(global_batch_size)
        # val_ds = tf.data.Dataset.from_tensor_slices((input_val, label_val)).batch(global_batch_size)

        # train_ds.prefetch(3)
        # val_ds.prefetch(3)

        train_ds = tf.data.Dataset.from_generator(DatasetGenerator('training.h5', global_batch_size),
                                                  output_signature=(tf.TensorSpec(shape=(global_batch_size, 32, 32, 18),
                                                                                  dtype=tf.float64),
                                                                    tf.TensorSpec(shape=17,
                                                                                  dtype=tf.int8)))
        val_ds = tf.data.Dataset.from_generator(DatasetGenerator('validation.h5', global_batch_size),
                                                output_signature=(tf.TensorSpec(shape=(global_batch_size, 32, 32, 18),
                                                                                dtype=tf.float64),
                                                                  tf.TensorSpec(shape=17,
                                                                                dtype=tf.int8)))

        model = compile_model(input_shape, num_classes, loss_function, optimizer)

        history = model.fit(train_ds,
                            epochs=num_epochs,
                            verbose=verbosity,
                            validation_data=val_ds)

    num_samples = 3  # bullshit

    throughput_metric.track((num_samples / num_workers) * num_epochs)
    latency_metric.track((num_samples / num_workers) * num_epochs)

    return {"model": model, "classifier": optimizer,
            "accuracy": history.history["accuracy"]}
