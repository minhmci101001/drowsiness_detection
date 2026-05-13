"""
core/landmark_extractor.py
===========================
Trích xuất 478 facial landmarks từ MediaPipe Face Landmarker (tasks API).
Tính thêm head pose (pitch/yaw/roll) từ các landmark chính.

Ghi chú về API mediapipe:
  mediapipe >= 0.10.30 đã xóa mp.solutions, dùng mp.tasks.vision.FaceLandmarker.
  Cần download model 'face_landmarker.task' (float16) trước khi dùng.

Output chính:
  - landmarks: dict[int → (x, y, z, visibility)]
  - EAR trái, EAR phải, EAR trung bình
  - MAR (miệng)
  - Head pose: pitch, yaw, roll (degree)
  - Visibility trung bình của landmark mắt
"""

import cv2
import numpy as np
import urllib.request
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging
import config

_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "face_landmarker.task")
_MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def _ensure_model():
    """Tải model nếu chưa có."""
    if os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 1_000_000:
        return
    print(f"[LandmarkExtractor] Downloading face_landmarker.task → {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print(f"[LandmarkExtractor] Download done ({os.path.getsize(_MODEL_PATH):,} bytes)")


@dataclass
class LandmarkResult:
    # Raw landmarks: index → (x_norm, y_norm, z_norm, visibility)
    landmarks: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)

    # Metrics tính từ landmark
    ear_left:   float = 0.0
    ear_right:  float = 0.0
    ear_avg:    float = 0.0
    mar:        float = 0.0

    # Head pose (degree)
    pitch: float = 0.0   # gật đầu (+ = cúi xuống)
    yaw:   float = 0.0   # quay ngang (+ = quay phải)
    roll:  float = 0.0   # nghiêng đầu

    # Chất lượng
    eye_visibility: float = 1.0   # 0–1
    mouth_visibility: float = 1.0  # 0–1, thấp = có thể đeo khẩu trang
    eye_brightness_ratio: float = 1.0  # eye_brightness / face_brightness, thấp = kính râm
    is_valid: bool = False


class LandmarkExtractor:
    """
    Sử dụng MediaPipe FaceLandmarker (tasks API) để detect 478 landmarks,
    sau đó tính các metrics cần thiết.
    """

    # Camera matrix giả định cho head pose (chuẩn cho webcam 720p)
    _CAM_MATRIX = None
    _DIST_COEFFS = np.zeros((4, 1))

    # 3D model points của khuôn mặt chuẩn (mm) — dùng cho solvePnP
    _FACE_3D_POINTS = np.array([
        [0.0,    0.0,    0.0   ],   # Nose tip (landmark 1)
        [0.0,   -330.0, -65.0 ],   # Chin (152)
        [-225.0, 170.0, -135.0],   # Left eye corner (33)
        [225.0,  170.0, -135.0],   # Right eye corner (263)
        [-150.0,-150.0, -125.0],   # Left mouth corner (61)
        [150.0, -150.0, -125.0],   # Right mouth corner (291)
    ], dtype=np.float64)

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._landmarker = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            _ensure_model()
            import mediapipe as mp
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            options = mp_vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.6,
                min_face_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self.logger.info("MediaPipe FaceLandmarker (tasks API) loaded")
        except ImportError:
            self.logger.error("mediapipe not installed!")
            raise
        except Exception as e:
            self.logger.error(f"FaceLandmarker init failed: {e}")
            raise

    def extract(self, frame: np.ndarray, timestamp_ms: int = 0) -> LandmarkResult:
        """
        Trích xuất landmarks và tính metrics từ một frame BGR (mode VIDEO).
        Trả về LandmarkResult (is_valid=False nếu không detect được mặt).
        """
        result = LandmarkResult()
        h, w = frame.shape[:2]

        # Khởi tạo camera matrix theo kích thước frame
        if self._CAM_MATRIX is None or self._CAM_MATRIX[0, 2] != w / 2:
            self._CAM_MATRIX = np.array([
                [w, 0,   w / 2],
                [0, w,   h / 2],
                [0, 0,   1    ]
            ], dtype=np.float64)

        # Chuyển frame sang mediapipe Image format
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detection_result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

        if not detection_result.face_landmarks:
            return result

        face_lm_list = detection_result.face_landmarks[0]

        # ── Lưu tất cả landmarks ──────────────────────────────────────────
        for i, lm in enumerate(face_lm_list):
            # tasks API: lm.x, lm.y, lm.z, lm.visibility (hoặc 1.0 nếu không có)
            vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility is not None else 1.0
            result.landmarks[i] = (lm.x, lm.y, lm.z, vis)

        # ── Tính EAR ──────────────────────────────────────────────────────
        result.ear_left  = self._calc_ear(result.landmarks, config.LEFT_EYE,  w, h)
        result.ear_right = self._calc_ear(result.landmarks, config.RIGHT_EYE, w, h)
        result.ear_avg   = (result.ear_left + result.ear_right) / 2.0

        # ── Tính MAR ──────────────────────────────────────────────────────
        result.mar = self._calc_mar(result.landmarks, config.MOUTH, w, h)

        # ── Tính eye visibility (trung bình visibility của landmark mắt) ──
        eye_indices = config.LEFT_EYE + config.RIGHT_EYE
        visibilities = [result.landmarks[i][3] for i in eye_indices if i in result.landmarks]
        result.eye_visibility = float(np.mean(visibilities)) if visibilities else 0.0

        # ── Tính mouth visibility (Cải tiến 3b: khẩu trang) ───────────────
        mouth_vis = [result.landmarks[i][3] for i in config.MOUTH if i in result.landmarks]
        result.mouth_visibility = float(np.mean(mouth_vis)) if mouth_vis else 0.0

        # ── Tính eye brightness ratio (Cải tiến 3a: kính râm) ─────────────
        result.eye_brightness_ratio = self._calc_eye_brightness_ratio(
            frame, result.landmarks, w, h
        )

        # ── Tính head pose ───────────────────────────────────────────────────
        pose = self._calc_head_pose(result.landmarks, w, h)
        if pose:
            result.pitch, result.yaw, result.roll = pose

        result.is_valid = True
        return result

    # ────────────────────────────────────────────────────────────────────────
    # EAR calculation
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _calc_ear(
        landmarks: Dict,
        indices: list,
        w: int, h: int
    ) -> float:
        if not all(i in landmarks for i in indices):
            return 0.0

        def pt(i):
            lm = landmarks[i]
            return np.array([lm[0] * w, lm[1] * h])

        P1, P2, P3, P4, P5, P6 = [pt(i) for i in indices]

        A = np.linalg.norm(P2 - P6)
        B = np.linalg.norm(P3 - P5)
        C = np.linalg.norm(P1 - P4)

        if C < 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)

    # ────────────────────────────────────────────────────────────────────────
    # MAR calculation
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _calc_mar(
        landmarks: Dict,
        indices: list,
        w: int, h: int
    ) -> float:
        if not all(i in landmarks for i in indices):
            return 0.0

        def pt(i):
            lm = landmarks[i]
            return np.array([lm[0] * w, lm[1] * h])

        pts = [pt(i) for i in indices]
        A = np.linalg.norm(pts[1] - pts[7])
        B = np.linalg.norm(pts[2] - pts[6])
        C = np.linalg.norm(pts[3] - pts[5])
        D = np.linalg.norm(pts[0] - pts[4])

        if D < 1e-6:
            return 0.0
        return (A + B + C) / (3.0 * D)

    # ────────────────────────────────────────────────────────────────────────
    # Head pose estimation (solvePnP)
    # ────────────────────────────────────────────────────────────────────────
    def _calc_head_pose(
        self,
        landmarks: Dict,
        w: int, h: int
    ) -> Optional[Tuple[float, float, float]]:
        indices = config.HEAD_POSE_POINTS
        if not all(i in landmarks for i in indices):
            return None

        image_points = np.array([
            [landmarks[i][0] * w, landmarks[i][1] * h]
            for i in indices
        ], dtype=np.float64)

        success, rot_vec, _ = cv2.solvePnP(
            self._FACE_3D_POINTS,
            image_points,
            self._CAM_MATRIX,
            self._DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        sy = np.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2( rot_mat[2,1], rot_mat[2,2])
            y = np.arctan2(-rot_mat[2,0], sy)
            z = np.arctan2( rot_mat[1,0], rot_mat[0,0])
        else:
            x = np.arctan2(-rot_mat[1,2], rot_mat[1,1])
            y = np.arctan2(-rot_mat[2,0], sy)
            z = 0

        pitch = np.degrees(x)
        yaw   = np.degrees(y)
        roll  = np.degrees(z)

        # solvePnP đôi khi trả về nghiệm lật (~±180°) → normalize về [-90, 90]
        if pitch > 90:
            pitch = pitch - 180
        elif pitch < -90:
            pitch = pitch + 180

        # Clamp yaw/roll về range hợp lý
        if yaw > 90:
            yaw = yaw - 180
        elif yaw < -90:
            yaw = yaw + 180

        return pitch, yaw, roll


    # ────────────────────────────────────────────────────────────────────────
    # Eye brightness ratio (sunglasses detection)
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _calc_eye_brightness_ratio(
        frame: np.ndarray,
        landmarks: Dict,
        w: int, h: int
    ) -> float:
        """
        So sánh brightness vùng mắt vs vùng mặt.
        Trả về ratio (0–1+). Thấp = kính râm che mắt.
        """
        try:
            # Lấy bounding box vùng mắt từ landmark
            eye_indices = config.LEFT_EYE + config.RIGHT_EYE
            eye_pts = []
            for i in eye_indices:
                if i in landmarks:
                    lm = landmarks[i]
                    eye_pts.append((int(lm[0] * w), int(lm[1] * h)))

            if len(eye_pts) < 4:
                return 1.0

            xs = [p[0] for p in eye_pts]
            ys = [p[1] for p in eye_pts]
            # Mở rộng vùng mắt lên/xuống 1 chút
            pad_y = max(int((max(ys) - min(ys)) * 0.5), 5)
            ey1 = max(0, min(ys) - pad_y)
            ey2 = min(h, max(ys) + pad_y)
            ex1 = max(0, min(xs))
            ex2 = min(w, max(xs))

            if ey2 <= ey1 or ex2 <= ex1:
                return 1.0

            # Brightness vùng mắt (kênh V trong HSV)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            eye_region = hsv[ey1:ey2, ex1:ex2, 2]  # V channel
            eye_brightness = float(eye_region.mean()) if eye_region.size > 0 else 128.0

            # Brightness vùng mặt (dùng convex hull của tất cả landmark)
            all_pts = [(int(landmarks[i][0] * w), int(landmarks[i][1] * h))
                       for i in landmarks if i < 468]
            if len(all_pts) < 10:
                return 1.0
            fxs = [p[0] for p in all_pts]
            fys = [p[1] for p in all_pts]
            fy1 = max(0, min(fys))
            fy2 = min(h, max(fys))
            fx1 = max(0, min(fxs))
            fx2 = min(w, max(fxs))

            face_region = hsv[fy1:fy2, fx1:fx2, 2]
            face_brightness = float(face_region.mean()) if face_region.size > 0 else 128.0

            if face_brightness < 10:
                return 1.0  # quá tối, không đủ dữ liệu

            return eye_brightness / face_brightness

        except Exception:
            return 1.0  # lỗi → coi như bình thường

    def cleanup(self):
        if self._landmarker:
            self._landmarker.close()
