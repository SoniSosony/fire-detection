import cv2
import tensorflow as tf
from tensorflow import keras
from classes.BatchGenerator import BatchGenerator

TRAIN_DATASET_DIR = "E:/Datasets/Fire images/videos/batches/train"
VALIDATE_DATASET_DIR = "E:/Datasets/Fire images/videos/batches/validate"
COLOR_SPACE = cv2.COLOR_RGB2HLS

model = keras.models.Sequential()
model.add(keras.layers.TimeDistributed(
    keras.layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                        input_shape=(5, 224, 224, 3))
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

model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, activation='relu', return_sequences=False)))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer="adam",
    loss=tf.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.build((60, 5, 224, 224, 3))
print(model.summary())

train_batch_generator = BatchGenerator(TRAIN_DATASET_DIR, COLOR_SPACE)
validate_batch_generator = BatchGenerator(VALIDATE_DATASET_DIR, COLOR_SPACE)

model.fit(train_batch_generator,
          validation_data=validate_batch_generator,
          epochs=40,
          validation_steps=6
          )

model.save('./models/cnn_lstm/1/hls_1')
