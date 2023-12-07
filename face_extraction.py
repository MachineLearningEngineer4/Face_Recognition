import cv2
import glob
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

path = "" # Путь к папке, из которой нужно вырезать лица
img_number = 1

img_list = glob.glob(path)

for file in img_list:
    img = cv2.imread(file, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    arr = np.asarray(gray)
    faces = face_cascade.detectMultiScale(arr, 1.3, 5)
    try:
        for (x, y, w, h) in faces:
            roi_color = img[y:y+h, x:x+w]
        resized = cv2.resize(roi_color, (128, 128))
        print(resized)
        cv2.imwrite("extracted_faces" + str(img_number) + ".jpg", resized)
        print("done")
    except:
        print("No faces detected")

    img_number += 1