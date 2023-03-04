import os
import cv2

IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
DATASET_DIR = "E:/Datasets/Fire images/videos/cut"
DESTINATION_DIR = "E:/Datasets/Fire images/videos/resized"
CLASSES_LIST = ["fire", "no_fire"]

for class_name in CLASSES_LIST:
    files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
    for file_name in files_list:
        print(file_name)

        video_reader = cv2.VideoCapture(os.path.join(DATASET_DIR, class_name, file_name))
        video_fps = video_reader.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(os.path.join(DESTINATION_DIR, class_name, file_name),
                                       fourcc, video_fps, (IMAGE_WIDTH, IMAGE_HEIGHT), True)

        while True:
            ret, frame = video_reader.read()
            if ret:
                dim = (IMAGE_WIDTH, IMAGE_HEIGHT)
                rescaled_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                video_writer.write(rescaled_frame)
            else:
                break

        print("release")
        video_reader.release()
        video_writer.release()
        cv2.destroyAllWindows()
