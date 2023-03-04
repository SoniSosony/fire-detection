import os
import cv2
import tensorflow as tf
from tensorflow import keras
from classes.ImagesBatchGenerator import ImagesBatchGenerator
from keras.applications.resnet import ResNet50

from cnn_architectures.time_distributed.VGG16 import add_vgg16_layers


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_DATASET_DIR = "E:/Datasets/Fire images/images/batches/train"
VALIDATE_DATASET_DIR = "E:/Datasets/Fire images/images/batches/validate"
TEST_DATASET_DIR = "E:/Datasets/Fire images/images/batches/test"
MODEL_PATH = './models/cnn_xyz_1'
COLOR_SPACE = cv2.COLOR_RGB2XYZ

model = keras.models.Sequential()
model.add(
    keras.layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                        input_shape=(224, 224, 3))
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
model.add(keras.layers.Dense(30, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

train_batch_generator = ImagesBatchGenerator(TRAIN_DATASET_DIR, COLOR_SPACE)
# validate_batch_generator = ImagesBatchGenerator(VALIDATE_DATASET_DIR, COLOR_SPACE)
test_batch_generator = ImagesBatchGenerator(TEST_DATASET_DIR, COLOR_SPACE)

model.fit(train_batch_generator,
          steps_per_epoch=64,
          # validation_data=validate_batch_generator,
          epochs=1000,
          # validation_steps=2
          )

# model.save(MODEL_PATH)

# test_loss, test_acc = model.evaluate(test_batch_generator)
# print(test_acc)

# new_model = tf.keras.models.load_model('./models/cnn_1')
# test_loss, test_acc_1 = new_model.evaluate(test_batch_generator)
# print(test_acc_1)

