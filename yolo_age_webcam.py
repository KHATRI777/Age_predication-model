import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

def fake_age_estimation(face_img):
    h = face_img.shape[0]
    if h < 100:
        return "0-18"
    elif h < 200:
        return "18-30"
    elif h < 300:
        return "30-50"
    else:
        return "50+"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # YOLO class 0 = person (we treat as face proxy)
            if cls == 0 and conf > 0.5:

                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                age = fake_age_estimation(face)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"Person | Age: {age}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,255,255), 2)

    cv2.imshow("YOLO Face + Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()