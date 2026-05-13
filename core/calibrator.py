"""
core/calibrator.py
===================
Calibrate EAR baseline cá nhân hóa cho từng người dùng.

Quy trình:
  1. Thu thập EAR trong N giây đầu khi người dùng tỉnh táo
  2. Dùng MEDIAN (không phải mean) để loại bỏ ảnh hưởng của chớp mắt
  3. Tính threshold = ratio × baseline
  4. Lưu/load profile để không cần calibrate lại mỗi lần

Tại sao median tốt hơn mean:
  - Mỗi lần chớp mắt EAR về ~0 → kéo mean xuống giả tạo
  - Median bỏ qua outlier → baseline phản ánh đúng mắt lúc mở bình thường
"""

import numpy as np
import json
import os
from typing import Optional
import logging
import config


class CalibrationState:
    WAITING    = "waiting"       # chờ bắt đầu
    COLLECTING = "collecting"    # đang thu thập dữ liệu
    DONE       = "done"          # đã xong


class Calibrator:
    """
    Thu thập và tính EAR baseline cá nhân hóa.
    """

    PROFILE_PATH = "calibration_profile.json"

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.state  = CalibrationState.WAITING

        self._ear_samples = []
        self._target_samples = int(config.CALIBRATION_DURATION_SEC * config.CALIBRATION_FPS_ASSUME)

        # Kết quả sau calibration
        self.baseline:  Optional[float] = None
        self.threshold: Optional[float] = None
        self.eye_type:  str = "normal"   # "normal" / "narrow" / "very_narrow"
        self.is_calibrated = False

        # Thử load profile đã lưu
        self._try_load_profile()

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────

    def start(self):
        """Bắt đầu quá trình thu thập (gọi khi user sẵn sàng)."""
        self.state = CalibrationState.COLLECTING
        self._ear_samples = []
        self.logger.info(f"Calibration started — collecting {self._target_samples} samples "
                         f"({config.CALIBRATION_DURATION_SEC}s)")

    def feed(self, ear: float, quality_ok: bool) -> bool:
        """
        Đưa một giá trị EAR vào.
        Chỉ nhận nếu quality_ok=True (tránh sample xấu do ánh sáng/quay mặt).

        Returns True khi calibration hoàn tất.
        """
        if self.state != CalibrationState.COLLECTING:
            return False
        if not quality_ok or ear <= 0:
            return False

        self._ear_samples.append(ear)

        if len(self._ear_samples) >= self._target_samples:
            self._finalize()
            return True
        return False

    def get_progress(self) -> float:
        """Tiến độ thu thập (0.0 → 1.0)."""
        if self.state == CalibrationState.DONE:
            return 1.0
        return min(len(self._ear_samples) / self._target_samples, 1.0)

    def recalibrate(self):
        """Reset và bắt đầu lại."""
        self.state = CalibrationState.WAITING
        self.is_calibrated = False
        self.baseline = None
        self.threshold = None
        self._ear_samples = []
        self.logger.info("Calibration reset")

    # ────────────────────────────────────────────────────────────────────────
    # Internal
    # ────────────────────────────────────────────────────────────────────────

    def _finalize(self):
        samples = np.array(self._ear_samples)

        # Loại bỏ outlier thấp (chớp mắt): bỏ 10% thấp nhất
        lower_cut = np.percentile(samples, 10)
        clean_samples = samples[samples > lower_cut]

        if len(clean_samples) < 10:
            self.logger.warning("Not enough clean samples for calibration, using all")
            clean_samples = samples

        self.baseline = float(np.median(clean_samples))

        # ── Phân loại eye type dựa trên baseline ─────────────────────────
        self.eye_type = self._classify_eye_type(self.baseline)

        # Dùng threshold ratio từ profile tương ứng (adaptive)
        profile = config.EYE_PROFILES.get(self.eye_type, config.EYE_PROFILES["normal"])
        adaptive_ratio = profile["threshold_ratio"]
        self.threshold = self.baseline * adaptive_ratio

        self.state = CalibrationState.DONE
        self.is_calibrated = True

        self.logger.info(
            f"Calibration done: baseline={self.baseline:.4f}, "
            f"threshold={self.threshold:.4f}, "
            f"eye_type={self.eye_type} (ratio={adaptive_ratio}) "
            f"(from {len(clean_samples)} clean samples)"
        )

        # Lưu profile
        self._save_profile()

    @staticmethod
    def _classify_eye_type(baseline: float) -> str:
        """Phân loại mắt dựa trên EAR baseline."""
        if baseline >= config.EYE_TYPE_NORMAL_MIN:
            return "normal"
        elif baseline >= config.EYE_TYPE_NARROW_MIN:
            return "narrow"
        else:
            return "very_narrow"

    def _save_profile(self):
        try:
            profile = {
                "baseline": self.baseline,
                "threshold": self.threshold,
                "eye_type": self.eye_type,
                "sample_count": len(self._ear_samples),
            }
            with open(self.PROFILE_PATH, "w") as f:
                json.dump(profile, f, indent=2)
            self.logger.info(f"Calibration profile saved to {self.PROFILE_PATH}")
        except Exception as e:
            self.logger.warning(f"Could not save calibration profile: {e}")

    def _try_load_profile(self):
        if not os.path.exists(self.PROFILE_PATH):
            return
        try:
            with open(self.PROFILE_PATH) as f:
                profile = json.load(f)
            self.baseline  = profile["baseline"]
            self.threshold = profile["threshold"]
            self.eye_type  = profile.get("eye_type", "normal")
            self.state = CalibrationState.DONE
            self.is_calibrated = True
            self.logger.info(
                f"Loaded saved calibration: baseline={self.baseline:.4f}, "
                f"threshold={self.threshold:.4f}, eye_type={self.eye_type}"
            )
        except Exception as e:
            self.logger.warning(f"Could not load calibration profile: {e}")
