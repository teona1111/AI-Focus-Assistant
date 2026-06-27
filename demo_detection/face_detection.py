import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    "eye_model.h5",
    compile=False
)

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FACE_OUTLINE = [
    10, 338, 297, 332, 284, 251, 389,
    356, 454, 323, 361, 288, 397,
    365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172,
    58, 132, 93, 234, 127, 162,
    21, 54, 103, 67, 109
]

cap = cv2.VideoCapture(0)
closed_frames = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            face_points = []
            for idx in FACE_OUTLINE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                face_points.append((x, y))
            face_points = np.array(face_points, np.int32)
            cv2.polylines(frame, [face_points], True, (0, 255, 0), 1)
            left_eye_points = []

            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye_points.append((x, y))

            left_eye_points = np.array(left_eye_points, np.int32)
            cv2.polylines(frame, [left_eye_points], True, (255, 0, 0), 1)
            right_eye_points = []

            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye_points.append((x, y))

            right_eye_points = np.array(right_eye_points, np.int32)
            cv2.polylines(frame, [right_eye_points], True, (255, 0, 0), 1)
            lx, ly, lw_eye, lh_eye = cv2.boundingRect(left_eye_points)
            rx, ry, rw_eye, rh_eye = cv2.boundingRect(right_eye_points)
            left_eye_crop = frame[ly:ly + lh_eye, lx:lx + lw_eye]
            right_eye_crop = frame[ry:ry + rh_eye, rx:rx + rw_eye]

            try:

                left_eye_crop = cv2.resize(left_eye_crop, (64, 64))
                right_eye_crop = cv2.resize(right_eye_crop, (64, 64))
                left_eye_crop = left_eye_crop.astype("float32") / 255.0
                right_eye_crop = right_eye_crop.astype("float32") / 255.0
                left_eye_crop = np.expand_dims(left_eye_crop, axis=0)
                right_eye_crop = np.expand_dims(right_eye_crop, axis=0)
                left_prediction = model.predict(left_eye_crop, verbose=0)
                right_prediction = model.predict(right_eye_crop, verbose=0)
                left_score = left_prediction[0][0]
                right_score = right_prediction[0][0]
                average_score = (left_score + right_score) / 2

                if average_score > 0.35:
                    closed_frames += 1
                else:
                    closed_frames = 0

                if closed_frames >= 3:
                    final_result = "Closed"
                    color = (0, 0, 255)
                else:
                    final_result = "Opened"
                    color = (0, 255, 0)

                cv2.putText(
                    frame,
                    f"Result: {final_result}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

            except Exception as e:
                print(e)

    cv2.imshow("AI Focus Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()