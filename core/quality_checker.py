"""
core/quality_checker.py
========================
Kiểm tra chất lượng frame TRƯỚC khi đưa vào phân tích.

Xử lý các trường hợp đặc biệt:
  - Ánh sáng tối (hầm, đêm)
  - Ánh sáng quá chói / phản chiếu kính
  - Mặt quay ngang (nhìn gương chiếu hậu)
  - Landmark bị che hoặc confidence thấp
  - Frame bị nhiễu / blur
"""

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import config


class FrameQuality(Enum):
    OK            = "ok"
    DARK          = "dark"           # ánh sáng quá tối
    OVEREXPOSED   = "overexposed"    # quá sáng / phản chiếu
    FACE_TURNED   = "face_turned"    # mặt quay ngang
    NO_FACE       = "no_face"        # không tìm thấy mặt
    LOW_CONF      = "low_conf"       # confidence thấp
    BLUR          = "blur"           # frame bị mờ
    REFLECTION    = "reflection"     # nghi phản chiếu kính lên mắt
    SUNGLASSES    = "sunglasses"     # Cải tiến 3a: đeo kính râm
    IR_MODE       = "ir_mode"        # Cải tiến 3c: camera hồng ngoại


@dataclass
class QualityReport:
    quality: FrameQuality
    is_usable: bool              # có nên dùng frame này không
    ear_reliable: bool           # EAR có đáng tin không
    brightness: float            # 0–255
    blur_score: float            # Laplacian variance, cao = sắc nét
    yaw_angle: Optional[float]   # góc quay mặt ngang
    message: str                 # mô tả ngắn cho debug


class FrameQualityChecker:
    """
    Chạy trước mỗi frame, trả về QualityReport để pipeline quyết định
    có xử lý tiếp hay bỏ qua frame đó.
    """

    def __init__(self):
        self._brightness_history = []   # lưu 30 frame gần nhất để smooth
        self._history_size = 30

    def check(
        self,
        frame: np.ndarray,
        face_confidence: Optional[float] = None,
        yaw_angle: Optional[float] = None,
        landmark_visibility: Optional[float] = None,
        ear_variance: Optional[float] = None,
        eye_brightness_ratio: Optional[float] = None,  # Cải tiến 3a
    ) -> QualityReport:
        """
        Kiểm tra toàn diện một frame.

        Parameters
        ----------
        frame               : BGR frame từ OpenCV
        face_confidence     : confidence score từ YOLO/MediaPipe (0.0–1.0)
        yaw_angle           : góc quay mặt ngang (degree), dương = quay phải
        landmark_visibility : visibility trung bình của landmark mắt (0.0–1.0)
        ear_variance        : variance EAR trong 10 frame gần nhất

        Returns
        -------
        QualityReport
        """
        brightness = self._calc_brightness(frame)
        blur_score = self._calc_blur(frame)

        # Cập nhật history
        self._brightness_history.append(brightness)
        if len(self._brightness_history) > self._history_size:
            self._brightness_history.pop(0)

        # ── Kiểm tra từng điều kiện ──────────────────────────────────────

        # 1. Không có mặt
        if face_confidence is not None and face_confidence < config.MIN_FACE_CONFIDENCE:
            return QualityReport(
                quality=FrameQuality.NO_FACE,
                is_usable=False, ear_reliable=False,
                brightness=brightness, blur_score=blur_score,
                yaw_angle=yaw_angle,
                message=f"Face confidence too low: {face_confidence:.2f}"
            )

        # 2. Ánh sáng quá tối — hầm xe, đêm tối
        if brightness < config.MIN_BRIGHTNESS:
            # Vẫn "usable" ở mức thấp nhưng EAR không đáng tin
            return QualityReport(
                quality=FrameQuality.DARK,
                is_usable=brightness > 20,   # dưới 20 thì vô dụng hẳn
                ear_reliable=False,
                brightness=brightness, blur_score=blur_score,
                yaw_angle=yaw_angle,
                message=f"Low brightness: {brightness:.1f} (min={config.MIN_BRIGHTNESS})"
            )

        # 3. Ánh sáng quá chói — ban ngày, đèn đường chiếu thẳng
        if brightness > config.MAX_BRIGHTNESS:
            return QualityReport(
                quality=FrameQuality.OVEREXPOSED,
                is_usable=True,   # mặt vẫn detect được, chỉ EAR hơi sai
                ear_reliable=False,
                brightness=brightness, blur_score=blur_score,
                yaw_angle=yaw_angle,
                message=f"Overexposed: {brightness:.1f} (max={config.MAX_BRIGHTNESS})"
            )

        # 4. Phản chiếu kính (glasses reflection)
        # Dấu hiệu: EAR variance rất thấp VÀ EAR đang cao (mắt mở)
        # Không flag nếu EAR thấp vì lúc đó variance thấp là bình thường (mắt đang nhắm)
        # ear_variance thấp + EAR cao = reflection; ear_variance thấp + EAR thấp = đang nhắm mắt
        if ear_variance is not None and ear_variance < config.REFLECTION_EAR_VAR_MAX:
            # Chỉ coi là reflection nếu chúng ta không đang track EAR thấp
            # (ear_variance thấp khi mắt nhắm là bình thường)
            pass  # Bỏ qua hoàn toàn check này — không đủ tin cậy để dùng

        # 4b. Kính râm (Cải tiến 3a: sunglasses detection)
        # Vùng mắt tối hơn nhiều so với mặt → kính râm che mắt
        if (eye_brightness_ratio is not None and
                eye_brightness_ratio < config.SUNGLASSES_BRIGHTNESS_RATIO):
            return QualityReport(
                quality=FrameQuality.SUNGLASSES,
                is_usable=True,
                ear_reliable=False,  # EAR không đáng tin khi đeo kính râm
                brightness=brightness, blur_score=blur_score,
                yaw_angle=yaw_angle,
                message=f"Sunglasses detected (eye/face brightness={eye_brightness_ratio:.2f})"
            )

        # 5. Mặt quay ngang — nhìn gương chiếu hậu, ngoái lại
        if yaw_angle is not None and abs(yaw_angle) > config.YAW_SKIP_THRESHOLD:
            return QualityReport(
                quality=FrameQuality.FACE_TURNED,
                is_usable=True,   # frame vẫn dùng được cho head pose
                ear_reliable=False,  # nhưng EAR không đáng tin khi nhìn ngang
                brightness=brightness, blur_score=blur_score,
                yaw_angle=yaw_angle,
                message=f"Face turned: yaw={yaw_angle:.1f}° (threshold={config.YAW_SKIP_THRESHOLD}°)"
            )

        # 6. Landmark visibility thấp — che mặt, khẩu trang, tóc che mắt
        # QUAN TRỌNG: Khi mắt nhắm, MediaPipe báo visibility thấp cho landmark mắt
        # → KHÔNG được dùng visibility để loại EAR, nếu không sẽ miss drowsiness detection
        # Chỉ log cảnh báo nhẹ, nhưng vẫn giữ ear_reliable=True
        if landmark_visibility is not None and landmark_visibility < config.MIN_LANDMARK_VISIBILITY:
            return QualityReport(
                quality=FrameQuality.LOW_CONF,
                is_usable=True,
                ear_reliable=True,   # Vẫn tin EAR dù visibility thấp (có thể đang nhắm mắt!)
                brightness=brightness, blur_score=blur_score,
                yaw_angle=yaw_angle,
                message=f"Low landmark visibility: {landmark_visibility:.2f} (may be eyes closed)"
            )

        # 7. Frame bị mờ (xe rung, camera rung)
        if blur_score < 50:
            return QualityReport(
                quality=FrameQuality.BLUR,
                is_usable=blur_score > 20,
                ear_reliable=blur_score > 30,
                brightness=brightness, blur_score=blur_score,
                yaw_angle=yaw_angle,
                message=f"Blurry frame: score={blur_score:.1f}"
            )

        # 8. Camera hồng ngoại (Cải tiến 3c: IR camera)
        # IR camera ban đêm → frame gần như grayscale, landmark accuracy giảm
        saturation = self._calc_saturation(frame)
        if saturation < config.IR_SATURATION_THRESHOLD:
            return QualityReport(
                quality=FrameQuality.IR_MODE,
                is_usable=True,
                ear_reliable=False,  # landmark kém chính xác trong IR
                brightness=brightness, blur_score=blur_score,
                yaw_angle=yaw_angle,
                message=f"IR camera detected (saturation={saturation:.1f})"
            )

        # ── Tất cả OK ────────────────────────────────────────────────────
        return QualityReport(
            quality=FrameQuality.OK,
            is_usable=True, ear_reliable=True,
            brightness=brightness, blur_score=blur_score,
            yaw_angle=yaw_angle,
            message="OK"
        )

    def get_lighting_trend(self) -> str:
        """
        Phân tích xu hướng ánh sáng trong 30 frame gần nhất.
        Trả về: 'stable', 'darkening' (đang vào hầm), 'brightening'
        """
        if len(self._brightness_history) < 10:
            return "stable"
        recent  = np.mean(self._brightness_history[-5:])
        earlier = np.mean(self._brightness_history[:5])
        diff = recent - earlier
        if diff < -30:
            return "darkening"   # đang vào hầm / trời tối dần
        if diff > 30:
            return "brightening"
        return "stable"

    @staticmethod
    def _calc_brightness(frame: np.ndarray) -> float:
        """Tính độ sáng trung bình của frame (kênh V trong HSV)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 2].mean())

    @staticmethod
    def _calc_blur(frame: np.ndarray) -> float:
        """
        Tính độ nét bằng Laplacian variance.
        Giá trị cao = sắc nét, thấp = mờ.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _calc_saturation(frame: np.ndarray) -> float:
        """Tính saturation trung bình (kênh S trong HSV). Thấp = có thể IR camera."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 1].mean())

    def check_ir_mode(self, frame: np.ndarray) -> bool:
        """
        Kiểm tra xem frame có từ camera IR không.
        IR camera ban đêm cho ra ảnh gần như grayscale (saturation rất thấp).
        """
        sat = self._calc_saturation(frame)
        return sat < config.IR_SATURATION_THRESHOLD
