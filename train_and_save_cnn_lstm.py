import os
import cv2
import tensorflow as tf
from tensorflow import keras
from classes.BatchGenerator import BatchGenerator

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_DATASET_DIR = "E:/Datasets/Fire images/videos/batches/train"
VALIDATE_DATASET_DIR = "E:/Datasets/Fire images/videos/batches/validate"


def _get_input_shape(color_space):
    if color_space == cv2.COLOR_RGB2GRAY:
        return 5, 224, 224, 1
    return 5, 224, 224, 3


def _get_input_shape_lstm(color_space):
    if color_space == cv2.COLOR_RGB2GRAY:
        return 60, 5, 224, 224, 1
    return 60, 5, 224, 224, 3


def train_and_save_cnn_lstm(color_space, model_path):
    model = keras.models.Sequential()
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                            input_shape=_get_input_shape(color_space))
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.BatchNormalization()
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.MaxPool2D(pool_size=(2, 2))
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same")
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.BatchNormalization()
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.MaxPool2D(pool_size=(3, 3))
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.BatchNormalization()
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same")
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.BatchNormalization()
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same")
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.BatchNormalization()
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.MaxPool2D(pool_size=(2, 2))
    ))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Flatten()
    ))

    model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, activation='tanh', return_sequences=False)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer="adam",
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    model.build(_get_input_shape_lstm(color_space))
    print(model.summary())

    train_batch_generator = BatchGenerator(TRAIN_DATASET_DIR, color_space)
    # validate_batch_generator = BatchGenerator(VALIDATE_DATASET_DIR, color_space)

    model.fit(train_batch_generator,
              # validation_data=validate_batch_generator,
              epochs=30,
              # validation_steps=6
              )

    model.save(model_path)
