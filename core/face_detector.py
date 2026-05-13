"""
core/face_detector.py
======================
Phát hiện khuôn mặt trong frame.

Chiến lược:
  - Ưu tiên dùng YOLO (nhanh, chính xác xa/gần)
  - Fallback sang OpenCV Haar Cascade nếu YOLO không available
    (mediapipe >= 0.10.30 đã xóa mp.solutions nên không dùng được)
  - Trả về bounding box chuẩn hóa để các module sau dùng
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging


@dataclass
class FaceDetection:
    x1: int; y1: int; x2: int; y2: int   # bounding box pixel coords
    confidence: float
    source: str   # "yolo" hoặc "haar"

    @property
    def width(self):  return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1
    @property
    def center(self): return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def as_crop(self, frame: np.ndarray, padding: float = 0.15) -> np.ndarray:
        """Crop vùng mặt ra khỏi frame, có thêm padding xung quanh."""
        h, w = frame.shape[:2]
        px = int(self.width  * padding)
        py = int(self.height * padding)
        x1 = max(0, self.x1 - px)
        y1 = max(0, self.y1 - py)
        x2 = min(w, self.x2 + px)
        y2 = min(h, self.y2 + py)
        return frame[y1:y2, x1:x2]


class FaceDetector:
    """
    Wrapper thống nhất cho YOLO và OpenCV Haar Cascade face detection.
    Tự động fallback nếu YOLO không available.
    """

    def __init__(self, use_yolo: bool = True, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.use_yolo = use_yolo
        self._yolo_model = None
        self._haar_cascade = None
        self._init_detectors()

    def _init_detectors(self):
        # ── YOLO ──────────────────────────────────────────────────────────
        if self.use_yolo:
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO("yolov8n-face.pt")
                self.logger.info("YOLO face detector loaded successfully")
            except ImportError:
                self.logger.warning("ultralytics not installed → falling back to Haar Cascade")
                self.use_yolo = False
            except Exception as e:
                self.logger.warning(f"YOLO load failed: {e} → falling back to Haar Cascade")
                self.use_yolo = False

        # ── OpenCV Haar Cascade fallback ──────────────────────────────────
        # mediapipe >= 0.10.30 đã xóa mp.solutions, dùng Haar Cascade thay thế
        try:
            self._haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self._haar_cascade.empty():
                self.logger.error("Haar Cascade file not found!")
                self._haar_cascade = None
            else:
                self.logger.info("OpenCV Haar Cascade face detector loaded successfully")
        except Exception as e:
            self.logger.error(f"Haar Cascade load failed: {e}")

    def detect(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """
        Phát hiện khuôn mặt lớn nhất trong frame.
        Trả về FaceDetection hoặc None nếu không tìm thấy.
        """
        if self.use_yolo and self._yolo_model is not None:
            result = self._detect_yolo(frame)
            if result is not None:
                return result
            self.logger.debug("YOLO no detection, trying Haar Cascade fallback")

        return self._detect_haar(frame)

    def _detect_yolo(self, frame: np.ndarray) -> Optional[FaceDetection]:
        try:
            results = self._yolo_model(frame, verbose=False, conf=0.5)
            if not results or len(results[0].boxes) == 0:
                return None

            boxes = results[0].boxes
            best_idx = int(boxes.conf.argmax())
            conf  = float(boxes.conf[best_idx])
            xyxy  = boxes.xyxy[best_idx].cpu().numpy().astype(int)

            return FaceDetection(
                x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                confidence=conf, source="yolo"
            )
        except Exception as e:
            self.logger.debug(f"YOLO detect error: {e}")
            return None

    def _detect_haar(self, frame: np.ndarray) -> Optional[FaceDetection]:
        if self._haar_cascade is None:
            return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self._haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) == 0:
                return None

            # Lấy face lớn nhất (diện tích lớn nhất)
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return FaceDetection(
                x1=x, y1=y, x2=x + w, y2=y + h,
                confidence=0.75,
                source="haar"
            )
        except Exception as e:
            self.logger.debug(f"Haar Cascade detect error: {e}")
            return None

    def cleanup(self):
        pass  # Haar Cascade không cần cleanup
