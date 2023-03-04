import os
import cv2


IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
SEQUENCE_LENGTH = 20
BATCH_SIZE = 8
DATASET_DIR = "E:/Datasets/Fire images/videos/resized"
DESTINATION_DIR = "E:/Datasets/Fire images/videos/cut_smaller"
CLASSES_LIST = ["fire", "no_fire"]

for class_name in CLASSES_LIST:
    files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
    video_count = 0
    for file_name in files_list:
        print(file_name)

        video_reader = cv2.VideoCapture(os.path.join(DATASET_DIR, class_name, file_name))
        video_fps = video_reader.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_writer = cv2.VideoWriter(os.path.join(DESTINATION_DIR, class_name, file_name),
        #                                fourcc, video_fps, (IMAGE_WIDTH, IMAGE_HEIGHT), True)

        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        video_split_count = video_frames_count // SEQUENCE_LENGTH
        video_frames_count = video_split_count * SEQUENCE_LENGTH

        video_writer = None
        frame_in_video_count = 0
        for frame_counter in range(video_frames_count):

            if video_writer is None:
                video_writer = cv2.VideoWriter(os.path.join(DESTINATION_DIR, class_name, f"{class_name}_{video_count}.mp4"),
                                               fourcc, video_fps, (IMAGE_WIDTH, IMAGE_HEIGHT), True)

            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
            success, frame = video_reader.read()
            if not success:
                break

            video_writer.write(frame)
            frame_in_video_count = frame_in_video_count + 1

            if frame_in_video_count == SEQUENCE_LENGTH:
                video_writer.release()
                video_writer = None
                video_count = video_count + 1
                frame_in_video_count = 0
                print(f"{class_name}_{video_count}.mp4")

        cv2.destroyAllWindows()
