"""
core/analyzer.py
=================
Engine phân tích buồn ngủ theo kiến trúc 3 tầng:

  Tầng 1 — EAR relative threshold (từ calibration)
  Tầng 2 — Thời gian liên tục + PERCLOS
  Tầng 3 — Weighted score từ nhiều tín hiệu

Output: DrowsinessLevel (AWAKE / WARNING / DROWSY / CRITICAL)
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Deque
import time
import logging
import config


class DrowsinessLevel(Enum):
    AWAKE    = 0
    WARNING  = 1   # dấu hiệu đầu tiên
    DROWSY   = 2   # cần cảnh báo
    CRITICAL = 3   # khẩn cấp


@dataclass
class AnalysisResult:
    level:          DrowsinessLevel = DrowsinessLevel.AWAKE
    score:          float = 0.0        # tổng score 0–1
    ear:            float = 0.0
    mar:            float = 0.0
    perclos:        float = 0.0        # 0–1
    pitch:          float = 0.0
    yaw:            float = 0.0

    # Flags từng tín hiệu
    ear_flag:       bool = False
    yaw_flag:       bool = False
    pitch_flag:     bool = False
    yawn_flag:      bool = False
    perclos_flag:   bool = False
    blink_flag:     bool = False   # blink rate quá thấp
    quick_drop_flag: bool = False  # EAR giảm đột ngột (velocity)

    # Blink stats
    blink_rate:           float = 0.0
    avg_blink_duration:   float = 0.0

    consec_frames:  int  = 0
    ear_reliable:   bool = True

    # Thông tin bổ sung cho display
    ear_baseline:   Optional[float] = None
    ear_threshold:  Optional[float] = None
    eye_type:       str = "normal"      # loại mắt hiện tại
    ear_velocity:   float = 0.0         # tốc độ thay đổi EAR


class DrowsinessAnalyzer:
    """
    Nhận metrics từ landmark extractor và quality checker,
    trả về mức độ buồn ngủ và các flags chi tiết.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # ── PERCLOS window ────────────────────────────────────────────────
        perclos_frames = int(config.PERCLOS_WINDOW_SEC * config.CALIBRATION_FPS_ASSUME)
        self._perclos_window: Deque[bool] = deque(maxlen=perclos_frames)

        # ── EAR smoothing ─────────────────────────────────────────────────
        self._ear_smoothed  = None
        self._mar_smoothed  = None

        # ── EAR variance (để detect phản chiếu kính) ─────────────────────
        self._ear_recent: Deque[float] = deque(maxlen=15)

        # ── EAR velocity (Cải tiến 2) ─────────────────────────────────────
        self._ear_velocity_window: Deque[float] = deque(maxlen=config.EAR_VELOCITY_WINDOW)

        # ── Consecutive counters ──────────────────────────────────────────
        self._ear_consec    = 0
        self._yawn_consec   = 0
        self._pitch_consec  = 0

        # ── Calibration ref ───────────────────────────────────────────────
        self._baseline:  Optional[float] = None
        self._threshold: Optional[float] = None

        # ── Adaptive eye type (Cải tiến 1) ────────────────────────────────
        self._eye_type = "normal"
        # Dynamic weights — được gán lại khi biết eye_type
        self._weights = config.EYE_PROFILES["normal"].copy()

    def set_eye_type(self, eye_type: str):
        """Điều chỉnh scoring weights theo loại mắt."""
        self._eye_type = eye_type
        profile = config.EYE_PROFILES.get(eye_type, config.EYE_PROFILES["normal"])
        self._weights = profile.copy()
        self.logger.info(
            f"Analyzer adapted for eye_type='{eye_type}': "
            f"EAR w={profile['score_weight_ear']}, "
            f"HeadPose w={profile['score_weight_head_pose']}, "
            f"consec={profile['ear_consec_frames']}f"
        )

    def set_calibration(self, baseline: float, threshold: float):
        self._baseline  = baseline
        self._threshold = threshold
        self.logger.info(f"Analyzer calibration set: baseline={baseline:.4f}, threshold={threshold:.4f}")

    def update(
        self,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        ear_reliable: bool,
        blink_flag: bool = False,
        blink_rate: float = 0.0,
        avg_blink_duration: float = 0.0,
    ) -> AnalysisResult:
        """
        Cập nhật một frame mới và trả về kết quả phân tích.

        Parameters
        ----------
        ear          : EAR trung bình (đã tính từ 2 mắt)
        mar          : MAR miệng
        pitch        : góc gật đầu (degree)
        yaw          : góc quay ngang (degree)
        ear_reliable : True nếu EAR đáng tin (từ QualityChecker)
        """
        result = AnalysisResult(
            ear=ear, mar=mar, pitch=pitch, yaw=yaw,
            ear_reliable=ear_reliable,
            ear_baseline=self._baseline,
            ear_threshold=self._threshold,
            blink_flag=blink_flag,
            blink_rate=blink_rate,
            avg_blink_duration=avg_blink_duration,
            eye_type=self._eye_type,
        )

        # ── Làm mượt EAR và MAR ──────────────────────────────────────────
        alpha_ear = config.EAR_SMOOTH_ALPHA
        alpha_mar = config.MAR_SMOOTH_ALPHA

        if self._ear_smoothed is None:
            self._ear_smoothed = ear
            self._mar_smoothed = mar
        else:
            self._ear_smoothed = alpha_ear * ear + (1 - alpha_ear) * self._ear_smoothed
            self._mar_smoothed = alpha_mar * mar + (1 - alpha_mar) * self._mar_smoothed

        smooth_ear = self._ear_smoothed

        # ── Lưu EAR vào buffer variance ──────────────────────────────────
        if ear_reliable:
            self._ear_recent.append(ear)
        result.ear = smooth_ear

        # ── EAR VELOCITY (Cải tiến 2) ────────────────────────────────────
        ear_velocity = 0.0
        quick_drop = False
        if ear_reliable:
            self._ear_velocity_window.append(smooth_ear)
            if len(self._ear_velocity_window) >= config.EAR_VELOCITY_WINDOW:
                oldest = self._ear_velocity_window[0]
                newest = self._ear_velocity_window[-1]
                ear_velocity = (newest - oldest) / len(self._ear_velocity_window)
                quick_drop = ear_velocity < config.EAR_VELOCITY_DROP_THRESH
        result.quick_drop_flag = quick_drop
        result.ear_velocity = ear_velocity

        # ── Lấy dynamic config từ eye profile ────────────────────────────
        w = self._weights
        ear_consec_limit = w.get("ear_consec_frames", config.EAR_CONSEC_FRAMES)

        # ── TẦNG 1: EAR Flag ─────────────────────────────────────────────
        eye_closed = False
        if ear_reliable and self._threshold is not None:
            eye_closed = smooth_ear < self._threshold
        elif ear_reliable:
            eye_closed = smooth_ear < config.EAR_HARD_MIN

        # PERCLOS: ghi nhận frame này
        self._perclos_window.append(eye_closed and ear_reliable)

        # Đếm consecutive EAR thấp
        if eye_closed and ear_reliable:
            self._ear_consec += 1
        elif ear_reliable and self._threshold is not None and smooth_ear > self._threshold * 1.2:
            # Mắt mở rõ ràng (EAR vượt 120% threshold) → reset ngay lập tức
            self._ear_consec = 0
        else:
            # Mắt mở nhưng EAR lương lưửng → giảm dần cảnh giác
            self._ear_consec = max(0, self._ear_consec - 4)

        result.ear_flag      = self._ear_consec >= ear_consec_limit
        result.consec_frames = self._ear_consec

        # ── TẦNG 2: PERCLOS ───────────────────────────────────────────────
        if len(self._perclos_window) > 30:
            closed_count = sum(self._perclos_window)
            result.perclos = closed_count / len(self._perclos_window)
        result.perclos_flag = result.perclos >= config.PERCLOS_ALERT_THRESHOLD

        # ── Head Pose Flags ───────────────────────────────────────────────
        result.yaw_flag = abs(yaw) > config.HEAD_YAW_THRESHOLD

        if pitch > config.HEAD_PITCH_THRESHOLD:
            self._pitch_consec += 1
        else:
            self._pitch_consec = max(0, self._pitch_consec - 2)
        result.pitch_flag = self._pitch_consec >= config.HEAD_POSE_CONSEC

        # ── Yawn Flag ─────────────────────────────────────────────────────
        if self._mar_smoothed > config.MAR_YAWN_THRESHOLD:
            self._yawn_consec += 1
        else:
            self._yawn_consec = max(0, self._yawn_consec - 1)
        result.yawn_flag = self._yawn_consec >= config.MAR_YAWN_CONSEC
        result.mar = self._mar_smoothed

        # ── TẦNG 3: Weighted Score (ADAPTIVE — Cải tiến 1) ───────────────
        score = 0.0

        if result.ear_flag and ear_reliable:
            score += w["score_weight_ear"]

        if result.pitch_flag:
            score += w["score_weight_head_pose"]

        if result.yawn_flag:
            score += w["score_weight_yawn"]

        if result.perclos_flag:
            perclos_ratio = min(result.perclos / config.PERCLOS_URGENT_THRESHOLD, 1.0)
            score += w["score_weight_perclos"] * perclos_ratio

        if blink_flag:
            score += w["score_weight_blink"]

        # EAR velocity quick drop (Cải tiến 2)
        if quick_drop and ear_reliable:
            score += config.SCORE_WEIGHT_VELOCITY

        result.score = min(score, 1.0)

        # ── Quyết định DrowsinessLevel ────────────────────────────────────
        if result.score >= config.SCORE_URGENT_THRESHOLD:
            result.level = DrowsinessLevel.CRITICAL
        elif result.score >= config.SCORE_ALERT_THRESHOLD:
            result.level = DrowsinessLevel.DROWSY
        elif result.ear_flag or result.perclos_flag or quick_drop:
            result.level = DrowsinessLevel.WARNING
        else:
            result.level = DrowsinessLevel.AWAKE

        return result

    def get_ear_variance(self) -> float:
        """EAR variance trong 15 frame gần nhất — dùng để detect phản chiếu kính."""
        if len(self._ear_recent) < 5:
            return 1.0   # không đủ data → không flag
        return float(np.var(list(self._ear_recent)))

    def reset(self):
        self._perclos_window.clear()
        self._ear_recent.clear()
        self._ear_velocity_window.clear()
        self._ear_smoothed = None
        self._mar_smoothed = None
        self._ear_consec   = 0
        self._yawn_consec  = 0
        self._pitch_consec = 0
