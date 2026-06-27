import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


LEFT_EYE = [33, 160, 158, 133, 153, 144]

RIGHT_EYE = [362, 385, 387, 263, 373, 380]


class EyePredictor:

    def __init__(self):

        self.model = tf.keras.models.load_model(
            "eye_model.h5",
            compile=False
        )

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(0)

        self.last_frame = None

    def update_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return

        frame = cv2.flip(
            frame,
            1
        )

        self.last_frame = frame

    def predict(self):

        if self.last_frame is None:
            return None

        frame = self.last_frame.copy()

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb
        )

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]

        left_points = []

        for idx in LEFT_EYE:

            x = int(
                face.landmark[idx].x * w
            )

            y = int(
                face.landmark[idx].y * h
            )

            left_points.append(
                (
                    x,
                    y
                )
            )

        right_points = []

        for idx in RIGHT_EYE:

            x = int(
                face.landmark[idx].x * w
            )

            y = int(
                face.landmark[idx].y * h
            )

            right_points.append(
                (
                    x,
                    y
                )
            )

        left_points = np.array(
            left_points,
            np.int32
        )

        right_points = np.array(
            right_points,
            np.int32
        )

        lx, ly, lw, lh = cv2.boundingRect(
            left_points
        )

        rx, ry, rw, rh = cv2.boundingRect(
            right_points
        )

        left_eye = frame[
            ly:ly + lh,
            lx:lx + lw
        ]

        right_eye = frame[
            ry:ry + rh,
            rx:rx + rw
        ]

        try:

            left_eye = cv2.resize(
                left_eye,
                (64, 64)
            )

            right_eye = cv2.resize(
                right_eye,
                (64, 64)
            )

        except:
            return None

        left_eye = (
            left_eye.astype(
                np.float32
            ) / 255.0
        )

        right_eye = (
            right_eye.astype(
                np.float32
            ) / 255.0
        )

        left_eye = np.expand_dims(
            left_eye,
            axis=0
        )

        right_eye = np.expand_dims(
            right_eye,
            axis=0
        )

        left_score = self.model.predict(
            left_eye,
            verbose=0
        )[0][0]

        right_score = self.model.predict(
            right_eye,
            verbose=0
        )[0][0]

        closed_prob = float(
            (left_score + right_score) / 2
        )

        open_prob = float(
            1 - closed_prob
        )

        return {

            "open": open_prob,

            "closed": closed_prob

        }

    def close(self):

        self.cap.release()


if __name__ == "__main__":

    predictor = EyePredictor()

    while True:

        predictor.update_frame()

        result = predictor.predict()

        print(result)

        if predictor.last_frame is not None:

            cv2.imshow(
                "Camera",
                predictor.last_frame
            )

        key = cv2.waitKey(1)

        if key == 27:
            break

    predictor.close()

    cv2.destroyAllWindows()