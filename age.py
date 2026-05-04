import cv2
import numpy as np

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"

face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)


def detect_faces(net, frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, boxes


def predict_age(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                 MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward()
    return age_list[preds[0].argmax()]


def process_image(image_path):
    frame = cv2.imread(image_path)

    if frame is None:
        print("Image not found")
        return

    frame, boxes = detect_faces(face_net, frame)

    for (x1, y1, x2, y2) in boxes:
        face = frame[max(0, y1-20):y2+20, max(0, x1-20):x2+20]

        if face.size == 0:
            continue

        age = predict_age(face)

        cv2.putText(frame, f"Age: {age}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

    cv2.imshow("Age Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


process_image("kid1.jpg")