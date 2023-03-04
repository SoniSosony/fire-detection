import os
import cv2
import numpy as np
from classes.FilesManager import FilesManager
from classes.Batch import Batch

IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
SEQUENCE_LENGTH = 5
BATCH_SIZE = 60
DATASET_DIR = "E:/Datasets/Fire images/videos/shuffled/test"
BATCHES_DIR = "E:/Datasets/Fire images/videos/batches/test"
CLASSES_LIST = ["fire", "no_fire"]


def generate_batch(video_path, class_name, batch, batches_count):
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    video_split_count = video_frames_count // SEQUENCE_LENGTH
    video_frames_count = video_split_count * SEQUENCE_LENGTH

    frames_batch = []
    for frame_counter in range(video_frames_count):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        success, frame = video_reader.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        frames_batch.append(resized_frame)

        if len(frames_batch) == SEQUENCE_LENGTH:
            batch.add(frames_batch, class_name)
            frames_batch = []

            if batch.length == BATCH_SIZE:
                save_batch(batch, batches_count)
                batch.clear()
                batches_count = batches_count + 1

            continue

    video_reader.release()
    return batches_count


def save_batch(batch, postfix):
    np_features = np.asarray(batch.features, dtype=np.uint8)
    np_labels = np.array(batch.labels)
    feature_batch_path = BATCHES_DIR + "/dataset_features_" + str(postfix)
    labels_batch_path = BATCHES_DIR + "/dataset_labels_" + str(postfix)
    np.savez_compressed(feature_batch_path, np_features)
    np.savez_compressed(labels_batch_path, np_labels)
    del np_features
    del np_labels


def create_dataset():
    files_manager = FilesManager(CLASSES_LIST, DATASET_DIR)
    batch = Batch()
    files_count = 0
    batches_count = 0

    while True:
        file_name, class_name = files_manager.get_unique_file()
        if file_name == -1:
            break

        video_files_path = os.path.join(DATASET_DIR, class_name, file_name)
        # print(f'Extracting Data of Class: "{class_name}" from "{file_name}"')

        batches_count = generate_batch(video_files_path, class_name, batch, batches_count)

        files_count = files_count + 1
        print(f'Files read: {files_count}')
