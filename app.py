import os

import cv2
from flask import Flask, Response, jsonify, render_template

from tracker import RallaTracker

VIDEO_PATH = os.environ.get("VIDEO_PATH", "res/Export 01-06-2026 10-18-50.MP4")
MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")
RALLA_CLASS_ID = int(os.environ.get("RALLA_CLASS_ID", "0"))
LINE_RATIO = float(os.environ.get("LINE_RATIO", "0.5"))

app = Flask(__name__)
tracker = RallaTracker(
    video_path=VIDEO_PATH,
    model_path=MODEL_PATH,
    ralla_class_id=RALLA_CLASS_ID,
    line_ratio=LINE_RATIO,
)


def generate_frames():
    while True:
        frame = tracker.process_next_frame()
        if frame is None:
            continue

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def stats():
    return jsonify(tracker.get_stats())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
