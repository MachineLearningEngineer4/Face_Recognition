import cv2
import glob

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

path = "" # Путь к папке, из которой нужно вырезать лица
img_number = 1

img_list = glob.glob(path)

for file in img_list:
    img = cv2.imread(file, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        resized = cv2.resize(gray, (128, 128))
        cv2.imwrite("bpm" + str(img_number) + ".jpg", resized)
        print("done")
        img_number += 1
    except:
        print("No faces detected")

