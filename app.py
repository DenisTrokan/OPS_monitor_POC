import os
import json

import cv2
from flask import Flask, Response, jsonify, render_template

from tracker import RallaTracker

VIDEO_PATH = os.environ.get("VIDEO_PATH", "res/Export 01-06-2026 10-18-50.MP4")
MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")
RALLA_CLASS_ID = int(os.environ.get("RALLA_CLASS_ID", "2"))
LINE_RATIO = float(os.environ.get("LINE_RATIO", "0.5"))
CONFIDENCE = float(os.environ.get("CONFIDENCE", "0.20"))
IMGSZ = int(os.environ.get("IMGSZ", "256 "))#da 416 default a 256 per performance migliore, ma se si vuole più precisione meglio 416 o 640  
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "6"))#2 = default, metto 4 per performance migliore

DEFAULT_ROI_POLYGON = [
    (0.50, 0.50),
    (0.71, 0.45),
    (0.77, 0.63),
    (0.57, 0.69),
]

ROI_POLYGON = DEFAULT_ROI_POLYGON
roi_polygon_raw = os.environ.get("ROI_POLYGON", "")
if roi_polygon_raw.strip():
    try:
        ROI_POLYGON = json.loads(roi_polygon_raw)
    except json.JSONDecodeError:
        ROI_POLYGON = DEFAULT_ROI_POLYGON

ROTATION_ANGLE = os.environ.get("ROTATION_ANGLE", "-45")
if ROTATION_ANGLE is not None and ROTATION_ANGLE != "":
    try:
        ROTATION_ANGLE = float(ROTATION_ANGLE)
    except ValueError:
        ROTATION_ANGLE = -45.0
else:
    ROTATION_ANGLE = -45.0

AUTO_ROTATE = os.environ.get("AUTO_ROTATE", "false").lower() in ("1", "true", "yes")

app = Flask(__name__)
tracker = RallaTracker(
    video_path=VIDEO_PATH,
    model_path=MODEL_PATH,
    ralla_class_id=RALLA_CLASS_ID,
    line_ratio=LINE_RATIO,
    conf=CONFIDENCE,
    imgsz=IMGSZ,
    rotation_angle=ROTATION_ANGLE,
    auto_rotate=AUTO_ROTATE,
    frame_stride=FRAME_STRIDE,
    roi_polygon=ROI_POLYGON,
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
