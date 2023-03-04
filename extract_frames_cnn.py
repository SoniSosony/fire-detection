import os
import cv2

IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
VIDEOS_DIR = "E:\Datasets\Fire images\\videos\shuffled\\test"
DESTINATION_DIR = "E:\Datasets\Fire images\images\images\\test"
CLASSES_LIST = ["fire", "no_fire"]


for class_name in CLASSES_LIST:
    files_list = os.listdir(os.path.join(VIDEOS_DIR, class_name))
    count = 0
    for file_name in files_list:
        print(file_name)
        video_reader = cv2.VideoCapture(os.path.join(VIDEOS_DIR, class_name, file_name))

        while True:
            ret, frame = video_reader.read()
            if ret:
                cv2.imwrite(os.path.join(DESTINATION_DIR, class_name, f'{class_name}_{count}.jpg'), frame)
                count = count + 1
            else:
                break

        video_reader.release()
        cv2.destroyAllWindows()
