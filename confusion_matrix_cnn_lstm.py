import os

import tensorflow as tf
import numpy as np
from classes.BatchGenerator import BatchGenerator
import cv2

RGB = "RGB"  # done
GRAY = cv2.COLOR_RGB2GRAY
XYZ = cv2.COLOR_RGB2XYZ  # done
CrCb = cv2.COLOR_RGB2YCrCb  # done
HSV = cv2.COLOR_RGB2HSV  # done
Lab = cv2.COLOR_RGB2Lab  # done
Luv = cv2.COLOR_RGB2Luv  # done
HLS = cv2.COLOR_RGB2HLS  # done
YUV = cv2.COLOR_RGB2YUV  # done

CNN_RGB_PATH = './models/cnn_lstm/1/rgb_1'
CNN_GRAY_PATH = './models/cnn_lstm/1/gray_1'
CNN_XYZ_PATH = './models/cnn_lstm/1/xyz_1'
CNN_CrCb_PATH = './models/cnn_lstm/1/CrCb_1'
CNN_HSV_PATH = './models/cnn_lstm/1/hsv_1'
CNN_Lab_PATH = './models/cnn_lstm/1/lab_1'
CNN_Luv_PATH = './models/cnn_lstm/1/luv_1'
CNN_HLS_PATH = './models/cnn_lstm/1/hls_1'
CNN_YUV_PATH = './models/cnn_lstm/1/yuv_1'

TEST_DATASET_DIR = "E:/Datasets/Fire images/videos/batches/test"
MODEL_DIR = './models_colab/cnn_lstm/1/lab_1'
COLOR_SPACE = Lab

batch_generator = BatchGenerator(TEST_DATASET_DIR, COLOR_SPACE)
# model = tf.keras.models.load_model(MODEL_DIR, custom_objects={"tf": tf})
model = tf.keras.models.load_model(MODEL_DIR, compile=False)
model.compile(
    optimizer="adam",
    loss=tf.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

x_train, y_train = [], []


def flatt_labels(labels, labels_to_flatt):
    for label_to_flatt in labels_to_flatt:
        labels.append(label_to_flatt)
    return labels


def predict_batches():
    labels = []
    predictions = []
    for i in range(batch_generator.len):
        images, classes = batch_generator.__getitem__(i)
        results = model.predict_on_batch(images)
        labels = flatt_labels(labels, classes.numpy())
        for result in results:
            predictions.append(np.argmax(result))
        print(f"Batches predicted: {i}")

    cm = tf.math.confusion_matrix(labels, predictions, 2)
    print(cm)

# print(labels)
# print(predictions)


# predict_batches()

test_loss, test_acc = model.evaluate(batch_generator)
print(test_acc)
