import os
import random
import shutil

DATA_DIR = "E:/Datasets/Fire images/videos/cut_smaller"
TRAIN_DIR = "E:/Datasets/Fire images/videos/shuffled/train"
VALIDATE_DIR = "E:/Datasets/Fire images/videos/shuffled/validate"
TEST_DIR = "E:/Datasets/Fire images/videos/shuffled/test"

FIRE = "fire"
NO_FIRE = "no_fire"


FIRE_TRAIN_MAX = 23865
FIRE_VALIDATE_MAX = 596
FIRE_TEST_MAX = 5369

NO_FIRE_TRAIN_MAX = 14750
NO_FIRE_VALIDATE_MAX = 368
NO_FIRE_TEST_MAX = 3320


def copy_files(file_list, src, dst, max_count):
    for i in range(max_count):
        shutil.copyfile(os.path.join(src, file_list[0]), os.path.join(dst, file_list[0]))
        del file_list[0]


fire_files = os.listdir(os.path.join(DATA_DIR, FIRE))
random.shuffle(fire_files)

copy_files(fire_files, os.path.join(DATA_DIR, FIRE), os.path.join(TRAIN_DIR, FIRE), FIRE_TRAIN_MAX)
copy_files(fire_files, os.path.join(DATA_DIR, FIRE), os.path.join(VALIDATE_DIR, FIRE), FIRE_VALIDATE_MAX)
copy_files(fire_files, os.path.join(DATA_DIR, FIRE), os.path.join(TEST_DIR, FIRE), FIRE_TEST_MAX)

fire_files = os.listdir(os.path.join(DATA_DIR, NO_FIRE))
random.shuffle(fire_files)

copy_files(fire_files, os.path.join(DATA_DIR, NO_FIRE), os.path.join(TRAIN_DIR, NO_FIRE), NO_FIRE_TRAIN_MAX)
copy_files(fire_files, os.path.join(DATA_DIR, NO_FIRE), os.path.join(VALIDATE_DIR, NO_FIRE), NO_FIRE_VALIDATE_MAX)
copy_files(fire_files, os.path.join(DATA_DIR, NO_FIRE), os.path.join(TEST_DIR, NO_FIRE), NO_FIRE_TEST_MAX)
