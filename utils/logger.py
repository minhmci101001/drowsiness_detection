"""
utils/logger.py
================
Logging setup và EventLogger (ghi log sự kiện buồn ngủ ra CSV).
"""

import logging
import csv
import os
import time
from datetime import datetime
from typing import Optional
from core.analyzer import AnalysisResult
from core.quality_checker import QualityReport


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("drowsiness")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class EventLogger:
    """
    Ghi log các sự kiện drowsiness vào file CSV.
    Hữu ích để phân tích sau và fine-tune thông số.
    """

    LOG_DIR = "logs"
    FIELDS  = [
        "timestamp", "level", "score",
        "ear", "ear_reliable", "ear_flag",
        "eye_type", "ear_velocity", "quick_drop_flag",
        "mar", "yawn_flag",
        "perclos", "perclos_flag",
        "pitch", "pitch_flag",
        "yaw", "yaw_flag",
        "blink_rate", "blink_flag", "avg_blink_duration_ms",
        "quality",
    ]

    def __init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(self.LOG_DIR, f"session_{ts}.csv")
        self._file = open(self._path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._last_log_time = 0.0

    def log_event(
        self,
        analysis: AnalysisResult,
        quality: Optional[QualityReport] = None,
        min_interval: float = 1.0   # ghi tối đa 1 dòng/giây để không spam
    ):
        now = time.time()
        if now - self._last_log_time < min_interval:
            return
        self._last_log_time = now

        self._writer.writerow({
            "timestamp":              datetime.now().isoformat(),
            "level":                  analysis.level.name,
            "score":                  f"{analysis.score:.3f}",
            "ear":                    f"{analysis.ear:.4f}",
            "ear_reliable":           analysis.ear_reliable,
            "ear_flag":               analysis.ear_flag,
            "eye_type":               getattr(analysis, 'eye_type', 'normal'),
            "ear_velocity":           f"{getattr(analysis, 'ear_velocity', 0.0):.5f}",
            "quick_drop_flag":        getattr(analysis, 'quick_drop_flag', False),
            "mar":                    f"{analysis.mar:.4f}",
            "yawn_flag":              analysis.yawn_flag,
            "perclos":                f"{analysis.perclos:.4f}",
            "perclos_flag":           analysis.perclos_flag,
            "pitch":                  f"{analysis.pitch:.2f}",
            "pitch_flag":             analysis.pitch_flag,
            "yaw":                    f"{analysis.yaw:.2f}",
            "yaw_flag":               analysis.yaw_flag,
            "blink_rate":             f"{analysis.blink_rate:.1f}",
            "blink_flag":             analysis.blink_flag,
            "avg_blink_duration_ms":  f"{analysis.avg_blink_duration:.0f}",
            "quality":                quality.quality.value if quality else "unknown",
        })
        self._file.flush()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()
            print(f"Event log saved: {self._path}")
