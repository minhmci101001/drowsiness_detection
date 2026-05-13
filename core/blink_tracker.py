"""
core/blink_tracker.py
======================
Theo dõi tần suất chớp mắt (blink rate) theo thời gian thực.

Tại sao blink rate quan trọng:
  - Người tỉnh táo chớp mắt ~15–20 lần/phút
  - Người buồn ngủ chớp mắt chậm lại: ~8–10 lần/phút
  - Người rất buồn ngủ: mắt nhắm lâu hơn giữa các lần chớp (slow blink)

Hai chỉ số được track:
  1. blink_rate   : số lần chớp/phút trong 1 phút gần nhất
  2. avg_blink_duration : thời gian trung bình mỗi lần nhắm (ms)
     → slow blink > 400ms là dấu hiệu buồn ngủ rõ ràng
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
import config


@dataclass
class BlinkStats:
    blink_rate:          float = 0.0   # lần/phút
    avg_blink_duration:  float = 0.0   # ms
    is_slow_blink:       bool  = False
    blink_flag:          bool  = False


class BlinkTracker:
    """
    State machine đơn giản để detect và đo blink:
      OPEN → CLOSING (EAR giảm xuống dưới threshold)
             → CLOSED (EAR < threshold liên tục)
             → OPENING (EAR tăng trở lại)
      OPEN (blink hoàn tất)
    """

    STATE_OPEN    = "open"
    STATE_CLOSING = "closing"
    STATE_CLOSED  = "closed"

    def __init__(self):
        self._state = self.STATE_OPEN
        self._blink_start: Optional[float] = None
        self._blink_timestamps = deque()
        self._blink_durations  = deque(maxlen=20)
        self._ear_threshold: Optional[float] = None

    def set_threshold(self, ear_threshold: float):
        """Nhận threshold từ calibrator để biết khi nào mắt đang nhắm."""
        self._ear_threshold = ear_threshold

    def update(self, ear: float, ear_reliable: bool) -> BlinkStats:
        """
        Cập nhật mỗi frame.
        Trả về BlinkStats hiện tại.
        """
        stats = BlinkStats()

        if not ear_reliable or self._ear_threshold is None:
            return stats

        now = time.time()
        is_closed = ear < self._ear_threshold

        # ── State machine ──────────────────────────────────────────────────
        if self._state == self.STATE_OPEN:
            if is_closed:
                self._state = self.STATE_CLOSING
                self._blink_start = now

        elif self._state == self.STATE_CLOSING:
            if is_closed:
                self._state = self.STATE_CLOSED
            else:
                # Mắt mở lại ngay → false blink (noise), reset
                self._state = self.STATE_OPEN
                self._blink_start = None

        elif self._state == self.STATE_CLOSED:
            if not is_closed:
                # Mắt vừa mở lại → blink hoàn tất
                duration_ms = (now - self._blink_start) * 1000 if self._blink_start else 0
                self._blink_start = None
                self._state = self.STATE_OPEN

                # Chỉ ghi nhận blink thật (50ms–2000ms)
                # Dưới 50ms = noise, trên 2s = đang ngủ hẳn (không phải blink)
                if 50 < duration_ms < 2000:
                    self._blink_timestamps.append(now)
                    self._blink_durations.append(duration_ms)

                stats.is_slow_blink = duration_ms > config.BLINK_SLOW_DURATION_MS

        # ── Dọn blink cũ hơn BLINK_WINDOW_SEC ────────────────────────────
        cutoff = now - config.BLINK_WINDOW_SEC
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        # ── Tính stats ────────────────────────────────────────────────────
        stats.blink_rate = len(self._blink_timestamps)   # số lần trong 60s = lần/phút

        if self._blink_durations:
            stats.avg_blink_duration = sum(self._blink_durations) / len(self._blink_durations)

        # Chỉ flag khi đã có đủ 30 giây dữ liệu (để tránh false flag lúc mới bật)
        has_enough_data = (
            len(self._blink_timestamps) >= 3 and
            now - (self._blink_timestamps[0] if self._blink_timestamps else now) >= 30
        )
        stats.blink_flag = has_enough_data and stats.blink_rate < config.BLINK_RATE_LOW_THRESHOLD

        return stats

    def reset(self):
        self._state = self.STATE_OPEN
        self._blink_start = None
        self._blink_timestamps.clear()
        self._blink_durations.clear()
