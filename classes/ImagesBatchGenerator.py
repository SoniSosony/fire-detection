from tensorflow import keras
import tensorflow as tf
import os
import numpy as np
import cv2

RGB = "RGB"


class ImagesBatchGenerator(keras.utils.Sequence):
    def __init__(self, files_path, color_space=RGB):
        self.files_path = files_path
        self.color_space = color_space
        self.files_list = os.listdir(self.files_path)
        self.len = int(len(self.files_list) / 2)
        self.classes = {"fire": 0, "no_fire": 1}

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        batch_x = np.load(os.path.join(self.files_path, self.files_list[idx]))['arr_0']
        batch_y = np.load(os.path.join(self.files_path, self.files_list[self.len + idx]))['arr_0']

        converted_batch_x = []
        if self.color_space != RGB:
            for i in range(batch_x.shape[0]):
                converted_batch_x.append(cv2.cvtColor(batch_x[i], self.color_space))
        else:
            converted_batch_x = batch_x

        batch_y = [0 if x == "fire" else 1 for x in batch_y]
        return tf.convert_to_tensor(np.array(converted_batch_x)), tf.convert_to_tensor(batch_y)
