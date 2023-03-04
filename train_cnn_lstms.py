import cv2
from train_and_save_cnn_lstm import train_and_save_cnn_lstm

RGB = "RGB"
GRAY = cv2.COLOR_RGB2GRAY
XYZ = cv2.COLOR_RGB2XYZ
CrCb = cv2.COLOR_RGB2YCrCb
HSV = cv2.COLOR_RGB2HSV
Lab = cv2.COLOR_RGB2Lab
Luv = cv2.COLOR_RGB2Luv
HLS = cv2.COLOR_RGB2HLS
YUV = cv2.COLOR_RGB2YUV

CNN_RGB_PATH = './models/cnn_lstm/1/rgb_1'
CNN_GRAY_PATH = './models/cnn_lstm/1/gray_1'
CNN_XYZ_PATH = './models/cnn_lstm/1/xyz_1'
CNN_CrCb_PATH = './models/cnn_lstm/1/CrCb_1'''
CNN_HSV_PATH = './models/cnn_lstm/1/hsv_1'
CNN_Lab_PATH = './models/cnn_lstm/1/lab_1'
CNN_Luv_PATH = './models/cnn_lstm/1/luv_1'
CNN_HLS_PATH = './models/cnn_lstm/1/hls_1'
CNN_YUV_PATH = './models/cnn_lstm/1/yuv_1'

# train_and_save_cnn_lstm(RGB, CNN_RGB_PATH)
# train_and_save_cnn_lstm(GRAY, CNN_GRAY_PATH)
# train_and_save_cnn_lstm(XYZ, CNN_XYZ_PATH)
train_and_save_cnn_lstm(CrCb, CNN_CrCb_PATH)
# train_and_save_cnn_lstm(HSV, CNN_HSV_PATH)
# train_and_save_cnn_lstm(Lab, CNN_Lab_PATH)
# train_and_save_cnn_lstm(Luv, CNN_Luv_PATH)
# train_and_save_cnn_lstm(HLS, CNN_HLS_PATH)
# train_and_save_cnn_lstm(YUV, CNN_YUV_PATH)
