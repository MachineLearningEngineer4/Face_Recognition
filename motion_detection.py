import datetime
import cv2
import time

cap = cv2.VideoCapture(0)
static_back = None
SECONDS_TO_RECORD_AFTER_DETECTION = 1
detection_stopped_time = None
detection = False
timer_started = False

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

    if static_back is None:
        static_back = gray
        continue

    diff_frame = cv2.absdiff(static_back, gray)

    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 1000:
            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 3)

    if len(cnts) > 0:
        if detection:
            timer_started = False
        else:
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            detection = cv2.VideoWriter(f"Datasets/{current_time}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (700, 500))
            print("start recording")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection.release()
                detection = False
                timer_started = False
                print("stop recording")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        detection.write(res)

    cv2.imshow('out', res)