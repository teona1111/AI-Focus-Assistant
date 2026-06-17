from flask import (
    Flask,
    jsonify,
    render_template,
    Response
)

import cv2
import threading
import time

from sensor_predictor import SensorPredictor
from eye_predictor import EyePredictor
from fusion import fuse_predictions


app = Flask(__name__)

sensor = SensorPredictor()

eye = EyePredictor()

latest_sensor = {
    "focus": 0.33,
    "sleepy": 0.33,
    "distracted": 0.33
}

latest_eye = {
    "open": 1.0,
    "closed": 0.0
}


def camera_loop():

    while True:

        try:

            eye.update_frame()

        except Exception as e:

            print(
                "Camera error:",
                e
            )


def eye_loop():

    global latest_eye

    while True:

        try:

            result = eye.predict()

            if result is not None:

                latest_eye = result

        except Exception as e:

            print(
                "Eye error:",
                e
            )

        time.sleep(1)


def sensor_loop():

    global latest_sensor

    while True:

        try:

            result = sensor.predict()

            if result is not None:

                latest_sensor = result

        except Exception as e:

            print(
                "Sensor error:",
                e
            )


threading.Thread(
    target=camera_loop,
    daemon=True
).start()

threading.Thread(
    target=eye_loop,
    daemon=True
).start()

threading.Thread(
    target=sensor_loop,
    daemon=True
).start()


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/status")
def status():

    try:

        fusion_result = fuse_predictions(
            latest_sensor,
            latest_eye
        )

        return jsonify({

            "sensor":
            latest_sensor,

            "eye":
            latest_eye,

            "final":
            fusion_result

        })

    except Exception as e:

        return jsonify({

            "error":
            str(e)

        })


@app.route("/video_feed")
def video_feed():

    def generate():

        while True:

            if eye.last_frame is None:

                continue

            ret, buffer = cv2.imencode(
                ".jpg",
                eye.last_frame
            )

            if not ret:

                continue

            frame = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )

    return Response(
        generate(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )