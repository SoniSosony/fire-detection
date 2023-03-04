import tensorflow as tf
import numpy as np
from classes.ImagesBatchGenerator import ImagesBatchGenerator
# from classes.BatchGenerator import BatchGenerator
import cv2

RGB = "RGB"
GRAY = cv2.COLOR_RGB2GRAY
XYZ = cv2.COLOR_RGB2XYZ
CrCb = cv2.COLOR_RGB2YCrCb
HSV = cv2.COLOR_RGB2HSV
Lab = cv2.COLOR_RGB2Lab
Luv = cv2.COLOR_RGB2Luv
HLS = cv2.COLOR_RGB2HLS
YUV = cv2.COLOR_RGB2YUV

CNN_RGB_PATH = './models/cnn/1/rgb_1'
CNN_GRAY_PATH = './models/cnn/1/gray_1'
CNN_XYZ_PATH = './models/cnn/1/xyz_1'
CNN_CrCb_PATH = './models/cnn/1/CrCb_1'
CNN_HSV_PATH = './models/cnn/1/hsv_1'
CNN_Lab_PATH = './models/cnn/1/lab_1'
CNN_Luv_PATH = './models/cnn/1/luv_1'
CNN_HLS_PATH = './models/cnn/1/hls_1'
CNN_YUV_PATH = './models/cnn/1/yuv_1'

TEST_DATASET_DIR = "E:/Datasets/Fire images/images/batches/test"
MODEL_DIR = CNN_GRAY_PATH
COLOR_SPACE = GRAY

batch_generator = ImagesBatchGenerator(TEST_DATASET_DIR, COLOR_SPACE)
model = tf.keras.models.load_model(MODEL_DIR, custom_objects={"tf": tf})

x_train, y_train = [], []

# labels = []
# predictions = []
# for i in range(batch_generator.len):
#     images, classes = batch_generator.__getitem__(i)
#     labels = [y for x in [labels, classes.numpy()] for y in x]
#     # results = model.predict(images, batch_size=300)
#     results = model.predict_on_batch(images)
#     for result in results:
#         predictions.append(np.argmax(result))
#     print(f"Batches predicted: {i}")
#
# cm = tf.math.confusion_matrix(labels, predictions, 2)
# print(cm)

test_loss, test_acc = model.evaluate(batch_generator)
print(test_acc)
