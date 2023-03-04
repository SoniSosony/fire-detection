import os
import random

IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
SEQUENCE_LENGTH = 360


class FilesManager:
    def __init__(self, classes_list, files_dir):
        self.read_files = {""}
        self.classes_list = classes_list
        self.files_dir = files_dir

        classes_0_length = len(os.listdir(os.path.join(files_dir, self.classes_list[0])))
        classes_1_length = len(os.listdir(os.path.join(files_dir, self.classes_list[1])))

        self.files_list = os.listdir(os.path.join(files_dir, self.classes_list[0]))
        self.files_list.extend(os.listdir(os.path.join(files_dir, self.classes_list[1])))

        self.files_classes = []
        self.files_classes += classes_0_length * [self.classes_list[0]]
        self.files_classes += classes_1_length * [self.classes_list[1]]

    def get_unique_file(self):
        while len(self.files_list) > 0:
            i = random.choice(range(len(self.files_list)))
            file_name = self.files_list.pop(i)
            class_name = self.files_classes.pop(i)
            return file_name, class_name

        return -1, -1
