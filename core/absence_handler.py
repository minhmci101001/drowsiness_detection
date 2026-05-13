"""
core/absence_handler.py
========================
Xử lý trường hợp mất mặt liên tục trong khung hình.

Các nguyên nhân có thể:
  1. Người lái quay hẳn ra ngoài lâu (nhìn điện thoại, nói chuyện)
  2. Gật đầu xuống (buồn ngủ nặng — đầu cúi thấp hơn camera)
  3. Che mặt / điều chỉnh kính
  4. Ánh sáng thay đổi đột ngột làm mất track

Hành vi:
  - Mất mặt < 2s    : bình thường (quay gương chiếu hậu, v.v.)
  - Mất mặt 2–5s    : WARNING — cảnh báo nhẹ
  - Mất mặt > 5s    : ALERT — có thể đang gật đầu/ngủ
  - Mất mặt > 10s   : CRITICAL — cảnh báo khẩn
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AbsenceLevel(Enum):
    OK       = "ok"
    SHORT    = "short"      # < 2s, bình thường
    WARNING  = "warning"    # 2–5s
    ALERT    = "alert"      # 5–10s
    CRITICAL = "critical"   # > 10s


@dataclass
class AbsenceReport:
    level:          AbsenceLevel = AbsenceLevel.OK
    duration_sec:   float = 0.0
    is_absent:      bool  = False
    should_alert:   bool  = False


class FaceAbsenceHandler:
    """
    Track thời gian liên tục không có mặt trong frame.
    Integrate vào pipeline để cảnh báo khi driver mất khỏi camera quá lâu.
    """

    WARN_SEC     = 2.0
    ALERT_SEC    = 5.0
    CRITICAL_SEC = 10.0

    def __init__(self):
        self._absent_since: Optional[float] = None
        self._last_seen: float = time.time()
        self._consecutive_absent = 0

    def update(self, face_detected: bool) -> AbsenceReport:
        """
        Gọi mỗi frame với kết quả face detection.
        """
        now = time.time()
        report = AbsenceReport()

        if face_detected:
            # Mặt xuất hiện → reset
            self._absent_since = None
            self._last_seen = now
            self._consecutive_absent = 0
            return report

        # Mặt không có
        report.is_absent = True
        if self._absent_since is None:
            self._absent_since = now

        duration = now - self._absent_since
        report.duration_sec = duration

        if duration < self.WARN_SEC:
            report.level = AbsenceLevel.SHORT
        elif duration < self.ALERT_SEC:
            report.level = AbsenceLevel.WARNING
            report.should_alert = True
        elif duration < self.CRITICAL_SEC:
            report.level = AbsenceLevel.ALERT
            report.should_alert = True
        else:
            report.level = AbsenceLevel.CRITICAL
            report.should_alert = True

        return report

    def get_time_since_last_seen(self) -> float:
        """Số giây từ lần cuối thấy mặt."""
        return time.time() - self._last_seen
