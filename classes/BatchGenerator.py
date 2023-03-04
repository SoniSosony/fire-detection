from tensorflow import keras
import tensorflow as tf
import os
import numpy as np
import cv2

RGB = "RGB"


class BatchGenerator(keras.utils.Sequence):
    def __init__(self, files_path, color_space):
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
        for i in range(batch_x.shape[0]):
            video = []
            for j in range(batch_x[i].shape[0]):
                if self.color_space != RGB:
                    video.append(cv2.cvtColor(batch_x[i][j], self.color_space))
                else:
                    video.append(batch_x[i][j])
            converted_batch_x.append(video)

        labels = []
        for label in batch_y:
            labels.append(self.classes[label])

        return tf.convert_to_tensor(np.array(converted_batch_x), dtype=tf.uint8), tf.convert_to_tensor(labels)
