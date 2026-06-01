import threading
from typing import Dict, Optional

import cv2
from ultralytics import YOLO


class RallaTracker:
    def __init__(
        self,
        video_path: str,
        model_path: str = "yolov8n.pt",
        ralla_class_id: int = 0,
        line_ratio: float = 0.5,
        conf: float = 0.25,
        imgsz: int = 640,
        max_missing: int = 60,
    ) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Impossibile aprire il video: {video_path}")

        self.model = YOLO(model_path)
        self.ralla_class_id = ralla_class_id
        self.line_ratio = line_ratio
        self.conf = conf
        self.imgsz = imgsz
        self.max_missing = max_missing

        self.line_y: Optional[int] = None
        self.imbarcati = 0
        self.sbarcati = 0
        self.last_direction = "N/D"

        self.frame_index = 0
        self.last_centroids: Dict[int, tuple[int, int]] = {}
        self.last_seen: Dict[int, int] = {}
        self.lock = threading.Lock()
        self.frame_lock = threading.Lock()

    def _read_frame(self):
        ok, frame = self.cap.read()
        if ok:
            return frame

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if ok:
            return frame

        return None

    def get_stats(self) -> Dict[str, object]:
        with self.lock:
            imbarcati = self.imbarcati
            sbarcati = self.sbarcati
            last_direction = self.last_direction

        return {
            "imbarcati": imbarcati,
            "sbarcati": sbarcati,
            "totale_movimentati": imbarcati + sbarcati,
            "ultima_direzione": last_direction,
        }

    def process_next_frame(self):
        with self.frame_lock:
            frame = self._read_frame()
            if frame is None:
                return None

        if self.line_y is None:
            # Imposta la linea virtuale in base all'altezza del frame (0.0-1.0).
            # Cambia line_ratio se il video ha una risoluzione diversa o vuoi spostarla.
            # Se preferisci un valore in pixel, puoi usare ad esempio line_y = 540 per un video 1080p.
            self.line_y = int(frame.shape[0] * self.line_ratio)

        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=[self.ralla_class_id],
            tracker="bytetrack.yaml",
            verbose=False,
        )

        self.frame_index += 1
        imbarco_delta = 0
        sbarco_delta = 0
        last_direction = None
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy
                if hasattr(xyxy, "cpu"):
                    xyxy = xyxy.cpu().numpy()

                ids = boxes.id
                if ids is not None and hasattr(ids, "cpu"):
                    ids = ids.cpu().numpy().astype(int)
                elif ids is None:
                    ids = [None] * len(xyxy)

                for (x1, y1, x2, y2), track_id in zip(xyxy, ids):
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    label = "Ralla"
                    if track_id is not None:
                        track_id = int(track_id)
                        label = f"Ralla #{track_id}"

                        prev = self.last_centroids.get(track_id)
                        if prev is not None:
                            prev_y = prev[1]
                            if prev_y < self.line_y <= cy:
                                sbarco_delta += 1
                                last_direction = "Sbarco"
                            elif prev_y > self.line_y >= cy:
                                imbarco_delta += 1
                                last_direction = "Imbarco"

                        self.last_centroids[track_id] = (cx, cy)
                        self.last_seen[track_id] = self.frame_index

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (46, 204, 113), 2)
                    cv2.circle(frame, (cx, cy), 3, (46, 204, 113), -1)
                    cv2.putText(
                        frame,
                        label,
                        (int(x1), max(20, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (235, 245, 255),
                        2,
                    )

        stale_ids = [
            track_id
            for track_id, last_seen in self.last_seen.items()
            if self.frame_index - last_seen > self.max_missing
        ]
        for track_id in stale_ids:
            self.last_seen.pop(track_id, None)
            self.last_centroids.pop(track_id, None)

        with self.lock:
            base_imbarcati = self.imbarcati
            base_sbarcati = self.sbarcati
            base_direction = self.last_direction

        display_imbarcati = base_imbarcati + imbarco_delta
        display_sbarcati = base_sbarcati + sbarco_delta
        display_total = display_imbarcati + display_sbarcati
        display_direction = last_direction or base_direction

        if imbarco_delta or sbarco_delta or last_direction:
            with self.lock:
                self.imbarcati += imbarco_delta
                self.sbarcati += sbarco_delta
                if last_direction:
                    self.last_direction = last_direction

        self._draw_overlay(frame, display_imbarcati, display_sbarcati, display_total, display_direction)
        return frame

    def _draw_overlay(
        self,
        frame,
        imbarcati: int,
        sbarcati: int,
        totale: int,
        last_direction: str,
    ) -> None:
        h, w = frame.shape[:2]
        line_y = self.line_y if self.line_y is not None else int(h * 0.5)

        cv2.line(frame, (0, line_y), (w, line_y), (0, 200, 255), 2)

        panel_x, panel_y = 16, 14
        panel_w, panel_h = 360, 110
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (18, 20, 26),
            -1,
        )

        cv2.putText(
            frame,
            f"Imbarcati: {imbarcati}",
            (panel_x + 12, panel_y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (236, 243, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Sbarcati: {sbarcati}",
            (panel_x + 12, panel_y + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (236, 243, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Totale: {totale}",
            (panel_x + 12, panel_y + 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (236, 243, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Ultima: {last_direction}",
            (panel_x + 180, panel_y + 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 214, 255),
            2,
        )
