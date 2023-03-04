import os

folder = "E:/Datasets/Fire images/videos/smaller_dataset/fire/"
for count, filename in enumerate(os.listdir(folder)):
    c = count + 0
    dst = f"fire {str(c)}.mp4"
    src =f"{folder}/{filename}"
    dst =f"{folder}/{dst}"
    os.rename(src, dst)