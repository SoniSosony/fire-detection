from classes.FilesManager import FilesManager
import os
import cv2
import numpy as np

DATASET_DIR = "E:/Datasets/Fire images/images/images/test"
BATCHES_DIR = "E:/Datasets/Fire images/images/batches/test"
CLASSES_LIST = ["fire", "no_fire"]
BATCH_SIZE = 300

files_manager = FilesManager(CLASSES_LIST, DATASET_DIR)

features = []
labels = []

count = 0
files_count = 0
batch_count = 0

while True:
    count = count + 1
    files_count = files_count + 1

    file_name, class_name = files_manager.get_unique_file()
    if file_name == -1:
        break

    img = cv2.imread(os.path.join(DATASET_DIR, class_name, file_name))
    # img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    features.append(img)
    labels.append(class_name)

    if count == BATCH_SIZE:
        np_features = np.asarray(features, dtype=np.uint8)
        np_labels = np.array(labels)
        features = []
        labels = []
        feature_batch_path = os.path.join(BATCHES_DIR, "dataset_features_" + str(batch_count))
        labels_batch_path = os.path.join(BATCHES_DIR, "dataset_labels_" + str(batch_count))
        np.savez_compressed(feature_batch_path, np_features)
        np.savez_compressed(labels_batch_path, np_labels)
        del np_features
        del np_labels
        batch_count = batch_count + 1
        count = 0
        print(f"Batch saved, image read: {files_count}")

# batch = np.load(os.path.join(BATCHES_DIR, "dataset_features_0.npz"))
# print(batch['arr_0'].shape)
#
# labels = np.load(os.path.join(BATCHES_DIR, "dataset_labels_0.npz"))
# print(labels['arr_0'])
# print(len(labels['arr_0']))
#
# cv2.imshow('image', batch['arr_0'][8])
# cv2.waitKey(0)

