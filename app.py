import os
import json

import cv2
from flask import Flask, Response, jsonify, render_template, request
import threading

from tracker import RallaTracker

app = Flask(__name__)

CONFIGS = {
    "dpi": {
        "name": "DPI - Dispositivi Protezione Individuale (DPI_video.mp4)",
        "video_path": "res/DPI_video.mp4",
        "model_path": "yolov8s-world.pt",
        "mode": "dpi",
        "ralla_class_id": 0,
        "line_ratio": 0.5,
        "conf": 0.20,
        "imgsz": 416,
        "rotation_angle": 0.0,
        "auto_rotate": False,
        "frame_stride": 3,
        "roi_polygon": None,
        "world_classes": ["person", "yellow vest"],
    },
    "sbarco_exit": {
        "name": "Sbarco/Imbarco - Uscita (exit.mp4)",
        "video_path": "res/exit.mp4",
        "model_path": "yolov8n.pt",
        "mode": "tugs",
        "ralla_class_id": 2,
        "line_ratio": 0.5,
        "conf": 0.20,
        "imgsz": 256,
        "rotation_angle": -45.0,
        "auto_rotate": False,
        "frame_stride": 6,
        "roi_polygon": [
            (0.50, 0.50),
            (0.71, 0.45),
            (0.77, 0.63),
            (0.57, 0.69),
        ],
        "world_classes": None,
    },
    "sbarco_entry": {
        "name": "Sbarco/Imbarco - Ingresso (entry.mp4)",
        "video_path": "res/entry.mp4",
        "model_path": "yolov8n.pt",
        "mode": "tugs",
        "ralla_class_id": 2,
        "line_ratio": 0.5,
        "conf": 0.20,
        "imgsz": 256,
        "rotation_angle": -45.0,
        "auto_rotate": False,
        "frame_stride": 6,
        "roi_polygon": [
            (0.50, 0.50),
            (0.71, 0.45),
            (0.77, 0.63),
            (0.57, 0.69),
        ],
        "world_classes": None,
    },
    "sbarco_export": {
        "name": "Sbarco/Imbarco - Export Completo",
        "video_path": "res/Export 01-06-2026 10-18-50.MP4",
        "model_path": "yolov8n.pt",
        "mode": "tugs",
        "ralla_class_id": 2,
        "line_ratio": 0.5,
        "conf": 0.20,
        "imgsz": 256,
        "rotation_angle": -45.0,
        "auto_rotate": False,
        "frame_stride": 6,
        "roi_polygon": [
            (0.50, 0.50),
            (0.71, 0.45),
            (0.77, 0.63),
            (0.57, 0.69),
        ],
        "world_classes": None,
    }
}

trackers = {}
trackers_lock = threading.Lock()

def get_tracker(config_key):
    if config_key not in CONFIGS:
        config_key = "sbarco_exit"
    
    with trackers_lock:
        if config_key not in trackers:
            cfg = CONFIGS[config_key]
            trackers[config_key] = RallaTracker(
                video_path=cfg["video_path"],
                model_path=cfg["model_path"],
                ralla_class_id=cfg["ralla_class_id"],
                line_ratio=cfg["line_ratio"],
                conf=cfg["conf"],
                imgsz=cfg["imgsz"],
                rotation_angle=cfg["rotation_angle"],
                auto_rotate=cfg["auto_rotate"],
                frame_stride=cfg["frame_stride"],
                roi_polygon=cfg["roi_polygon"],
                mode=cfg["mode"],
                world_classes=cfg["world_classes"],
            )
        return trackers[config_key]


def generate_frames(config_key):
    tracker = get_tracker(config_key)
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
    config_key = request.args.get("config", "sbarco_exit")
    return Response(
        generate_frames(config_key),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def stats():
    config_key = request.args.get("config", "sbarco_exit")
    tracker = get_tracker(config_key)
    return jsonify(tracker.get_stats())


@app.route("/api/configs")
def get_configs():
    return jsonify({k: v["name"] for k, v in CONFIGS.items()})


@app.route("/api/reset")
def reset_tracker():
    config_key = request.args.get("config", "sbarco_exit")
    tracker = get_tracker(config_key)
    tracker.reset_counters()
    return jsonify({"status": "ok", "stats": tracker.get_stats()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
