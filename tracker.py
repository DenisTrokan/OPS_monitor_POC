import threading
import time
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class RallaTracker:
    def __init__(
        self,
        video_path: str,
        model_path: str = "yolov8n.pt",
        ralla_class_id: int = 0,
        line_ratio: float = 0.5,
        conf: float = 0.20,
        imgsz: int = 416,
        max_missing: int = 60,
        rotation_angle: Optional[float] = None,
        auto_rotate: bool = False,
        frame_stride: int = 2,
        roi_polygon: Optional[Sequence[Sequence[float]]] = None,
        mode: str = "tugs",
        world_classes: Optional[list] = None,
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
        self.rotation_angle = rotation_angle
        self.auto_rotate = auto_rotate
        self.frame_stride = max(1, int(frame_stride))
        self.roi_polygon_raw = [tuple(point) for point in roi_polygon] if roi_polygon else None
        self.mode = mode
        
        if world_classes and ("world" in model_path.lower() or "world" in str(type(self.model.model)).lower()):
            try:
                self.model.set_classes(world_classes)
            except Exception as e:
                print(f"Warning: set_classes failed: {e}")

        self.imbarcati = 0
        self.sbarcati = 0
        self.current_con_dpi = 0
        self.current_senza_dpi = 0
        self.last_direction = "N/D"
        self.last_detection_count = 0
        self.active_tracks = 0
        self.fps = 0.0

        self.frame_index = 0
        self.last_seen: Dict[int, int] = {}
        self.track_states: Dict[int, Dict[str, object]] = {}
        self.lock = threading.Lock()
        self.frame_lock = threading.Lock()

    def _read_frame(self):
        frame = None
        for _ in range(self.frame_stride):
            ok, current_frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, current_frame = self.cap.read()
                if not ok:
                    return None
            frame = current_frame
        return frame

    def _resolve_roi_polygon(self, frame) -> Optional[np.ndarray]:
        if not self.roi_polygon_raw:
            return None

        height, width = frame.shape[:2]
        points = np.asarray(self.roi_polygon_raw, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            return None

        if np.max(points) <= 1.5:
            points[:, 0] *= width
            points[:, 1] *= height

        return np.round(points).astype(np.int32)

    @staticmethod
    def _point_in_roi(roi_polygon: Optional[np.ndarray], point: Tuple[int, int]) -> bool:
        if roi_polygon is None or len(roi_polygon) < 3:
            return True
        return cv2.pointPolygonTest(roi_polygon, point, False) >= 0

    @staticmethod
    def _classify_side(roi_polygon: Optional[np.ndarray], x_coord: int) -> str:
        if roi_polygon is None or len(roi_polygon) < 3:
            return "center"
        center_x = float(np.mean(roi_polygon[:, 0]))
        return "right" if x_coord >= center_x else "left"

    def get_stats(self) -> Dict[str, object]:
        with self.lock:
            stats = {
                "mode": self.mode,
                "fps": round(self.fps, 2),
                "detections_roi": self.last_detection_count,
                "active_tracks": self.active_tracks,
            }
            if self.mode == "dpi":
                stats["con_dpi"] = self.current_con_dpi
                stats["senza_dpi"] = self.current_senza_dpi
                stats["totale"] = self.current_con_dpi + self.current_senza_dpi
            else:
                stats["imbarcati"] = self.imbarcati
                stats["sbarcati"] = self.sbarcati
                stats["totale_movimentati"] = self.imbarcati + self.sbarcati
                stats["ultima_direzione"] = self.last_direction
            return stats

    def reset_counters(self) -> None:
        with self.lock:
            self.imbarcati = 0
            self.sbarcati = 0
            self.current_con_dpi = 0
            self.current_senza_dpi = 0
            self.last_direction = "N/D"
            self.last_detection_count = 0
            self.active_tracks = 0
            self.fps = 0.0

        with self.frame_lock:
            self.frame_index = 0
            self.last_seen.clear()
            self.track_states.clear()
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def process_next_frame(self):
        with self.frame_lock:
            frame = self._read_frame()
            if frame is None:
                return None

        if self.rotation_angle is None and self.auto_rotate and self.frame_index == 0:
            try:
                estimated_angle = self._estimate_rotation_angle(frame)
                self.rotation_angle = -estimated_angle if estimated_angle is not None else 0.0
            except Exception:
                self.rotation_angle = 0.0

        if self.rotation_angle is not None and abs(self.rotation_angle) > 0.05:
            frame = self._rotate_image(frame, self.rotation_angle)

        roi_polygon = self._resolve_roi_polygon(frame)
        if self.mode == "dpi":
            roi_polygon = None
        roi_crop = frame
        roi_offset_x = 0
        roi_offset_y = 0

        if roi_polygon is not None and len(roi_polygon) >= 3:
            x, y, w, h = cv2.boundingRect(roi_polygon)
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x + max(1, w))
            y2 = min(frame.shape[0], y + max(1, h))
            roi_crop = frame[y:y2, x:x2]
            roi_offset_x = x
            roi_offset_y = y

        start_time = time.perf_counter()
        results = self.model.track(
            roi_crop,
            persist=True,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=[self.ralla_class_id] if self.mode == "tugs" else None,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        elapsed = max(time.perf_counter() - start_time, 1e-6)

        self.frame_index += 1
        imbarco_delta = 0
        sbarco_delta = 0
        con_dpi_count = 0
        senza_dpi_count = 0
        detections_in_roi = 0
        active_tracks = 0
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

                confs = boxes.conf
                if confs is not None and hasattr(confs, "cpu"):
                    confs = confs.cpu().numpy()
                elif confs is None:
                    confs = [None] * len(xyxy)

                clss = boxes.cls
                if clss is not None and hasattr(clss, "cpu"):
                    clss = clss.cpu().numpy().astype(int)
                elif clss is None:
                    clss = [0] * len(xyxy)

                if self.mode == "dpi":
                    people = []
                    vests = []
                    for (x1, y1, x2, y2), track_id, confidence, cls_id in zip(xyxy, ids, confs, clss):
                        if cls_id == 0:
                            people.append((x1, y1, x2, y2, confidence, track_id))
                        elif cls_id == 1:
                            vests.append((x1, y1, x2, y2, confidence))
                    
                    for px1, py1, px2, py2, pconf, track_id in people:
                        has_vest = False
                        for vx1, vy1, vx2, vy2, vconf in vests:
                            vcx = (vx1 + vx2) / 2
                            vcy = (vy1 + vy2) / 2
                            if px1 <= vcx <= px2 and py1 <= vcy <= py2:
                                has_vest = True
                                break
                        
                        x1_full = int(float(px1) + roi_offset_x)
                        y1_full = int(float(py1) + roi_offset_y)
                        x2_full = int(float(px2) + roi_offset_x)
                        y2_full = int(float(py2) + roi_offset_y)
                        cx = int((x1_full + x2_full) / 2)
                        cy = int((y1_full + y2_full) / 2)
                        
                        if track_id is not None:
                            self.last_seen[int(track_id)] = self.frame_index
                            active_tracks += 1
                        
                        inside_roi = self._point_in_roi(roi_polygon, (cx, cy))
                        if inside_roi:
                            detections_in_roi += 1

                        if has_vest:
                            con_dpi_count += 1
                            box_color = (0, 200, 0)
                            label = "DPI OK"
                        else:
                            senza_dpi_count += 1
                            box_color = (0, 0, 255)
                            label = "NO DPI"
                        
                        cv2.rectangle(frame, (x1_full, y1_full), (x2_full, y2_full), box_color, 2)
                        cv2.putText(frame, f"{label} {float(pconf):.2f}", (x1_full, max(20, y1_full - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                else:
                    for (x1, y1, x2, y2), track_id, confidence in zip(xyxy, ids, confs):
                        x1_full = int(float(x1) + roi_offset_x)
                        y1_full = int(float(y1) + roi_offset_y)
                        x2_full = int(float(x2) + roi_offset_x)
                        y2_full = int(float(y2) + roi_offset_y)
                        cx = int((x1_full + x2_full) / 2)
                        cy = int((y1_full + y2_full) / 2)
    
                        inside_roi = self._point_in_roi(roi_polygon, (cx, cy))
                        if inside_roi:
                            detections_in_roi += 1
    
                        # median X of ROI for debug/side classification
                        roi_center_x = None
                        if roi_polygon is not None and len(roi_polygon) >= 3:
                            roi_center_x = int(float(np.mean(roi_polygon[:, 0])))
    
                        label = "Ralla"
                        box_color = (46, 204, 113) if inside_roi else (120, 120, 120)
    
                        if track_id is not None:
                            track_id = int(track_id)
                            active_tracks += 1
                            label = f"Ralla #{track_id}"
    
                            state = self.track_states.setdefault(
                                track_id,
                                {
                                    "inside_roi": False,
                                    "entry_x": None,
                                    "last_centroid": None,
                                },
                            )
    
                            state["last_centroid"] = (cx, cy)
                            self.last_seen[track_id] = self.frame_index
    
                            if inside_roi and not state["inside_roi"]:
                                state["inside_roi"] = True
                                state["entry_x"] = cx
                            elif not inside_roi and state["inside_roi"] and state["entry_x"] is not None:
                                entry_x = float(state["entry_x"])
                                if cx < entry_x:
                                    sbarco_delta += 1
                                    last_direction = "Sbarco"
                                else:
                                    imbarco_delta += 1
                                    last_direction = "Imbarco"
                                state["inside_roi"] = False
                                state["entry_x"] = None
    
                        cv2.rectangle(frame, (x1_full, y1_full), (x2_full, y2_full), box_color, 2)
                        cv2.circle(frame, (cx, cy), 3, box_color, -1)
    
                        label_text = label
                        if confidence is not None:
                            label_text = f"{label} {float(confidence):.2f}"
                        cv2.putText(
                            frame,
                            label_text,
                            (x1_full, max(20, y1_full - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (235, 245, 255),
                            2,
                        )
    
                        # Debug overlay: show entry/exit side near the box
                        try:
                            state_dbg = self.track_states.get(track_id)
                            dbg_text = ""
                            if state_dbg is not None:
                                entry_x = state_dbg.get("entry_x")
                                if entry_x is not None and roi_center_x is not None:
                                    entry_side = "R" if entry_x >= roi_center_x else "L"
                                    exit_side = "R" if cx >= roi_center_x else "L"
                                    dbg_text = f"E:{entry_side}->X:{exit_side}"
                            if dbg_text:
                                cv2.putText(frame, dbg_text, (x1_full, min(frame.shape[0]-6, y2_full + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 100), 2)
                        except Exception:
                            pass


        stale_ids = [
            track_id
            for track_id, last_seen in self.last_seen.items()
            if self.frame_index - last_seen > self.max_missing
        ]
        for track_id in stale_ids:
            self.last_seen.pop(track_id, None)
            self.track_states.pop(track_id, None)

        with self.lock:
            if self.mode == "dpi":
                self.current_con_dpi = con_dpi_count
                self.current_senza_dpi = senza_dpi_count
            else:
                if imbarco_delta or sbarco_delta or last_direction:
                    self.imbarcati += imbarco_delta
                    self.sbarcati += sbarco_delta
                    if last_direction:
                        self.last_direction = last_direction

            self.fps = 1.0 / elapsed
            self.last_detection_count = detections_in_roi
            self.active_tracks = active_tracks
            display_fps = self.fps
            display_detections = self.last_detection_count
            display_tracks = self.active_tracks
            
            if self.mode == "dpi":
                display_imbarcati = self.current_con_dpi
                display_sbarcati = self.current_senza_dpi
                display_total = self.current_con_dpi + self.current_senza_dpi
                display_direction = "N/A"
            else:
                display_imbarcati = self.imbarcati
                display_sbarcati = self.sbarcati
                display_total = self.imbarcati + self.sbarcati
                display_direction = self.last_direction

        self._draw_overlay(
            frame,
            display_imbarcati,
            display_sbarcati,
            display_total,
            display_direction,
            roi_polygon,
            display_fps,
            display_detections,
            display_tracks,
        )
        return frame

    def _draw_overlay(
        self,
        frame,
        imbarcati: int,
        sbarcati: int,
        totale: int,
        last_direction: str,
        roi_polygon: Optional[np.ndarray],
        fps: float,
        detections_in_roi: int,
        active_tracks: int,
    ) -> None:
        if roi_polygon is not None and len(roi_polygon) >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [roi_polygon], (0, 0, 255))
            frame[:] = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
            cv2.polylines(frame, [roi_polygon], True, (0, 0, 255), 3)
            roi_text_x = int(np.min(roi_polygon[:, 0]))
            roi_text_y = max(24, int(np.min(roi_polygon[:, 1])) - 10)
            cv2.putText(frame, "ZONA CONTEGGIO", (roi_text_x, roi_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            # draw ROI median X line for diagnostics
            try:
                roi_cx = int(float(np.mean(roi_polygon[:, 0])))
                y_min = int(np.min(roi_polygon[:, 1]))
                y_max = int(np.max(roi_polygon[:, 1]))
                cv2.line(frame, (roi_cx, y_min), (roi_cx, y_max), (255, 220, 100), 2)
                cv2.putText(frame, "MID", (roi_cx + 6, max(18, y_min + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 100), 2)
            except Exception:
                pass

        panel_x, panel_y = 16, 14
        panel_w, panel_h = 430, 124
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 20, 26), -1)

        if self.mode == "dpi":
            cv2.putText(frame, f"Con DPI (OK): {imbarcati}", (panel_x + 12, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            cv2.putText(frame, f"Senza DPI (NO): {sbarcati}", (panel_x + 12, panel_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Totale in Frame: {totale}", (panel_x + 12, panel_y + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (236, 243, 255), 2)
            cv2.putText(frame, f"Mode: {self.mode.upper()}", (panel_x + 12, panel_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (120, 214, 255), 2)
        else:
            cv2.putText(frame, f"Imbarcati: {imbarcati}", (panel_x + 12, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (236, 243, 255), 2)
            cv2.putText(frame, f"Sbarcati: {sbarcati}", (panel_x + 12, panel_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (236, 243, 255), 2)
            cv2.putText(frame, f"Totale: {totale}", (panel_x + 12, panel_y + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (236, 243, 255), 2)
            cv2.putText(frame, f"Ultima: {last_direction}", (panel_x + 12, panel_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (120, 214, 255), 2)

        status_x = max(12, frame.shape[1] - 300)
        status_y = 18
        cv2.rectangle(frame, (status_x, status_y), (frame.shape[1] - 16, status_y + 116), (18, 20, 26), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (status_x + 12, status_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Det in ROI: {detections_in_roi}", (status_x + 12, status_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracks: {active_tracks}", (status_x + 12, status_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(frame, f"Conf: {self.conf:.2f}", (status_x + 12, status_y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 230, 140), 2)

    def _rotate_image(self, image, angle_deg: float):
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _estimate_rotation_angle(self, frame, debug: bool = False) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=20)
        angles = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                angle = float(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
                if abs(angle) < 45:
                    angles.append(angle)

        if not angles:
            return 0.0
        return float(np.median(angles))