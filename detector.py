import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOVideoDetector:
    """Captures frames with OpenCV, runs a YOLO model, and keeps the latest annotated frame."""

    def __init__(self, model_name: str = "yolov8n.pt", source=0, conf: float = 0.25,
                 iou: float = 0.45, img_size: int = 640):
        self.model = YOLO(model_name)
        self.source = source
        self.conf = conf
        self.iou = iou
        self.img_size = img_size

        self._cap = None
        self._thread = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None      # annotated frame (BGR)
        self._latest_detections = []   # list of dicts
        self._fps = 0.0

    # ---------- lifecycle ----------
    def start(self):
        if self._running.is_set():
            return
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # ---------- processing loop ----------
    def _loop(self):
        last_time = time.time()
        while self._running.is_set():
            ok, frame = self._cap.read()
            if not ok:
                break

            results = self.model.predict(
                frame, conf=self.conf, iou=self.iou,
                imgsz=self.img_size, verbose=False
            )[0]

            annotated = results.plot()  # draws boxes + labels
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "label": self.model.names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": [int(v) for v in box.xyxy[0].tolist()],
                })

            now = time.time()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            with self._lock:
                self._latest_frame = annotated
                self._latest_detections = detections
                self._fps = fps

        self._running.clear()

    # ---------- accessors for the UI ----------
    def get_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def get_detections(self):
        with self._lock:
            return list(self._latest_detections)

    def get_fps(self) -> float:
        with self._lock:
            return self._fps