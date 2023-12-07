import datetime
import cv2
import time

cap = cv2.VideoCapture(0)
static_back = None
SECONDS_TO_RECORD_AFTER_DETECTION = 1
detection_stopped_time = None
detection = False
timer_started = False
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("start stream")

while True:
    check, frame = cap.read()

    if frame is None:
        print("No frame")
        cap.stop()

    k = cv2.waitKey(1) & 0xFF

    res = cv2.resize(frame, (700, 500))

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, width, height) in faces:
        cv2.rectangle(res, (x, y), (x + width, y + height), (0, 255, 0), 2)

    if len(faces) > 0:
        current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        cv2.imwrite(f"Datasets/{current_time}.png", res)
        if detection:
            timer_started = False
        else:
            detection = cv2.VideoWriter(f"Datasets/{current_time}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (700, 500))
            print("started recording")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection.release()
                detection = False
                timer_started = False
                print("stopped recording")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        detection.write(res)

    cv2.imshow('out', res)