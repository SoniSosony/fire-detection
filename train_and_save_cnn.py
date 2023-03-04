import os
import cv2
import tensorflow as tf
from tensorflow import keras
from classes.ImagesBatchGenerator import ImagesBatchGenerator

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_DATASET_DIR = "E:/Datasets/Fire images/images/batches/train"
VALIDATE_DATASET_DIR = "E:/Datasets/Fire images/images/batches/validate"


def _get_input_shape(color_space):
    if color_space == cv2.COLOR_RGB2GRAY:
        return 224, 224, 1
    return 224, 224, 3


def train_and_save_cnn(color_space, model_path):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                            input_shape=_get_input_shape(color_space))
    )
    model.add(
        keras.layers.BatchNormalization()
    )
    model.add(
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))
    )
    model.add(
        keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same")
    )
    model.add(
        keras.layers.BatchNormalization()
    )
    model.add(
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))
    )
    model.add(
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")
    )
    model.add(
        keras.layers.BatchNormalization()
    )
    model.add(
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")
    )
    model.add(
        keras.layers.BatchNormalization()
    )
    model.add(
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")
    )
    model.add(
        keras.layers.BatchNormalization()
    )
    model.add(
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))
    )
    model.add(
        keras.layers.Flatten()
    )
    # model.add(keras.layers.Dense(30, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(1024, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer="adam",
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    model.summary()

    train_batch_generator = ImagesBatchGenerator(TRAIN_DATASET_DIR, color_space)
    validate_batch_generator = ImagesBatchGenerator(VALIDATE_DATASET_DIR, color_space)

    model.fit(train_batch_generator,
              # steps_per_epoch=64,
              # validation_data=validate_batch_generator,
              epochs=30,
              # validation_steps=2
              )

    model.save(model_path)
