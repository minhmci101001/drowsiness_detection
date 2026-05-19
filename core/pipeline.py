"""
core/pipeline.py
=================
Pipeline chính: kết nối tất cả modules và chạy vòng lặp video.

Luồng xử lý mỗi frame:
  1. Đọc frame từ camera
  2. FaceDetector → tìm vị trí khuôn mặt (YOLO / MediaPipe)
  3. LandmarkExtractor → 478 points + EAR/MAR/head pose
  4. QualityChecker → kiểm tra chất lượng frame
  5. Calibrator → nếu chưa calibrate, thu thập EAR baseline
  6. DrowsinessAnalyzer → tính score và level
  7. AlertSystem → phát cảnh báo nếu cần
  8. DisplayRenderer → vẽ HUD lên frame
  9. Hiển thị + log
"""

import cv2
import time
import logging
import numpy as np
from collections import deque
from typing import Optional

from core.face_detector      import FaceDetector
from core.landmark_extractor import LandmarkExtractor
from core.quality_checker    import FrameQualityChecker, FrameQuality
from core.calibrator         import Calibrator, CalibrationState
from core.analyzer           import DrowsinessAnalyzer, DrowsinessLevel
from core.blink_tracker      import BlinkTracker
from core.absence_handler    import FaceAbsenceHandler, AbsenceLevel
from core.alert              import AlertSystem
from core.display            import DisplayRenderer
from utils.logger            import EventLogger
import config


class DrowsinessPipeline:

    def __init__(self, use_yolo: bool = True, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initializing pipeline modules...")

        self.face_detector   = FaceDetector(use_yolo=use_yolo, logger=self.logger)
        self.landmark_ext    = LandmarkExtractor(logger=self.logger)
        self.quality_checker = FrameQualityChecker()
        self.calibrator      = Calibrator(logger=self.logger)
        self.analyzer        = DrowsinessAnalyzer(logger=self.logger)
        self.blink_tracker   = BlinkTracker()
        self.absence_handler = FaceAbsenceHandler()
        self.alert           = AlertSystem(logger=self.logger)
        self.display         = DisplayRenderer()
        self.event_logger    = EventLogger()

        # FPS tracking
        self._fps_times: deque = deque(maxlen=30)
        self._fps = 0.0
        self._session_start = time.time()
        self._alert_count = 0

        # Sync calibrator → analyzer nếu đã load profile
        if self.calibrator.is_calibrated:
            self.analyzer.set_calibration(
                self.calibrator.baseline,
                self.calibrator.threshold
            )
            self.analyzer.set_eye_type(self.calibrator.eye_type)
            self.blink_tracker.set_threshold(self.calibrator.threshold)

        # IR mode / System tracking
        self._ir_mode = False
        self._ir_check_interval = 90   # kiểm tra IR mỗi 3 giây (@ 30fps)
        self._frame_counter = 0        # Dùng chung cho IR và YOLO skip logic
        self._last_face = None

        self.logger.info("Pipeline ready.")

    def run(self, cap: cv2.VideoCapture):
        """Vòng lặp chính."""
        screenshot_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                elapsed = time.time() - self._session_start
                self.logger.info(
                    f"Stream ended — session {elapsed:.0f}s, "
                    f"alerts triggered: {self._alert_count}"
                )
                break

            # Flip ngang — camera thường bị ngược chiều, flip để hiển thị đúng
            frame = cv2.flip(frame, 1)

            # ── FPS ──────────────────────────────────────────────────────────────
            self._fps = self._calc_fps()

            # ── 1. Face detection (Cải tiến 5: Skip frame) ─────────────────────
            self._frame_counter += 1

            # Skip YOLO chỉ khi đã có mặt; nếu mất mặt thì detect mỗi frame để recovery nhanh
            should_detect = (
                self._last_face is None or                          # chưa có / mất mặt
                self._frame_counter % config.FACE_DETECTION_SKIP == 1  # frame detect định kỳ
            )
            if should_detect:
                face = self.face_detector.detect(frame)
                self._last_face = face
            else:
                face = self._last_face

            # ── 2. Landmark extraction (Cải tiến 4: VIDEO mode) ───────────
            lm_result = None
            if face is not None:
                # Truyền timestamp cho MediaPipe VIDEO mode
                timestamp_ms = int(time.monotonic() * 1000)
                lm_result = self.landmark_ext.extract(frame, timestamp_ms=timestamp_ms)

            # ── 2b. IR mode check (mỗi vài giây) ─────────────────────────
            if self._frame_counter % self._ir_check_interval == 0:
                ir_now = self.quality_checker.check_ir_mode(frame)
                if ir_now != self._ir_mode:
                    self._ir_mode = ir_now
                    if ir_now:
                        self.logger.info("IR camera mode detected — accuracy may be reduced")
                    else:
                        self.logger.info("IR camera mode ended — normal mode")

            # ── 3. Quality check ──────────────────────────────────────
            ear_variance = self.analyzer.get_ear_variance()
            quality = self.quality_checker.check(
                frame=frame,
                face_confidence=face.confidence if face else 0.0,
                yaw_angle=lm_result.yaw if lm_result and lm_result.is_valid else None,
                landmark_visibility=lm_result.eye_visibility if lm_result and lm_result.is_valid else None,
                ear_variance=ear_variance,
                eye_brightness_ratio=lm_result.eye_brightness_ratio if lm_result and lm_result.is_valid else None,
            )

            # ── 4. Face absence tracking ──────────────────────────────────
            absence = self.absence_handler.update(face_detected=(face is not None))
            if absence.should_alert:
                # Mất mặt lâu → cảnh báo riêng (có thể gật đầu ngủ)
                self.alert.process_absence(absence.level)
                self.logger.debug(f"Face absent {absence.duration_sec:.1f}s — {absence.level.value}")

            # ── 5. Calibration ────────────────────────────────────────────
            analysis = None
            if lm_result and lm_result.is_valid:
                ear_ok = quality.ear_reliable

                # Auto-start calibration nếu chưa calibrate và mặt đã ổn định
                if (not self.calibrator.is_calibrated and
                        self.calibrator.state == CalibrationState.WAITING and
                        quality.quality.value == "ok"):
                    self.calibrator.start()
                    self.logger.info("Auto-started calibration (stable face detected)")

                if not self.calibrator.is_calibrated:
                    done = self.calibrator.feed(lm_result.ear_avg, ear_ok)
                    if done:
                        self.analyzer.set_calibration(
                            self.calibrator.baseline,
                            self.calibrator.threshold
                        )
                        self.analyzer.set_eye_type(self.calibrator.eye_type)
                        self.blink_tracker.set_threshold(self.calibrator.threshold)
                        # Phát âm thanh báo calibration xong
                        self.alert._play_sound(config.ALERT_SOUND_CALIB_DONE, repeat=1)

                # ── 6. Blink tracking ─────────────────────────────────────
                blink_stats = self.blink_tracker.update(
                    ear=lm_result.ear_avg,
                    ear_reliable=ear_ok,
                )

                # ── 7. Analysis ─────────────────────────────────────────
                # Kiểm tra khẩu trang (Cải tiến 3b): bỏ qua MAR nếu mouth bị che
                effective_mar = lm_result.mar
                if lm_result.mouth_visibility < config.MOUTH_VISIBILITY_MIN:
                    effective_mar = 0.0   # không tính yawn khi đeo khẩu trang

                if self.calibrator.is_calibrated:
                    analysis = self.analyzer.update(
                        ear=lm_result.ear_avg,
                        mar=effective_mar,
                        pitch=lm_result.pitch,
                        yaw=lm_result.yaw,
                        ear_reliable=ear_ok,
                        blink_flag=blink_stats.blink_flag,
                        blink_rate=blink_stats.blink_rate,
                        avg_blink_duration=blink_stats.avg_blink_duration,
                    )

                    # ── 8. Alert ──────────────────────────────────────────
                    # Chỉ phát cảnh báo khi mắt thực sự đang nhắm (ear_flag=True)
                    # PERCLOS / pitch cao nhưng mắt đang mở → không kêu liên tục
                    if analysis.ear_flag:
                        self.alert.process(analysis.level)
                        if analysis.level != DrowsinessLevel.AWAKE:
                            self._alert_count += 1
                    else:
                        self.alert.process(DrowsinessLevel.AWAKE)

                    # ── 9. Log sự kiện quan trọng ─────────────────────────
                    if analysis.level != DrowsinessLevel.AWAKE:
                        self.event_logger.log_event(analysis, quality)

            # ── 10. Render ────────────────────────────────────────────────
            landmarks_dict = lm_result.landmarks if (lm_result and lm_result.is_valid) else None
            rendered = self.display.render(
                frame=frame,
                analysis=analysis,
                quality=quality if lm_result else None,
                calibrator=self.calibrator,
                landmarks=landmarks_dict,
                fps=self._fps,
                absence_report=absence,
            )

            # Vẽ bounding box mặt
            if face is not None:
                color = (0, 255, 0) if quality.is_usable else (0, 165, 255)
                cv2.rectangle(rendered, (face.x1, face.y1), (face.x2, face.y2), color, 2)
                cv2.putText(rendered, f"{face.source} {face.confidence:.2f}",
                            (face.x1, face.y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Absence warning đã được vẽ bên trong render() — không gọi lại ở đây

            cv2.imshow("Drowsiness Detection", rendered)

            # ── Keyboard ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.logger.info("User quit")
                break
            elif key == ord('r'):
                # Recalibrate: reset tất cả state liên quan
                self.calibrator.recalibrate()
                self.analyzer.reset()
                self.blink_tracker.reset()
                self.logger.info("Recalibration triggered — nhìn thẳng camera")
            elif key == ord('c'):
                # Manual override: force start calibration
                if not self.calibrator.is_calibrated:
                    self.calibrator.start()
                else:
                    self.logger.info("Already calibrated — press 'R' to recalibrate")
            elif key == ord('s'):
                fname = f"screenshot_{screenshot_idx:04d}.jpg"
                cv2.imwrite(fname, rendered)
                self.logger.info(f"Screenshot saved: {fname}")
                screenshot_idx += 1
            elif key == ord('d'):
                config.SHOW_DEBUG_INFO = not config.SHOW_DEBUG_INFO
                config.SHOW_LANDMARKS  = not config.SHOW_LANDMARKS

    def _calc_fps(self) -> float:
        now = time.time()
        self._fps_times.append(now)
        if len(self._fps_times) < 2:
            return 0.0
        return (len(self._fps_times) - 1) / (self._fps_times[-1] - self._fps_times[0])

    def cleanup(self):
        self.face_detector.cleanup()
        self.landmark_ext.cleanup()
        self.alert.cleanup()
        self.event_logger.close()
        self.blink_tracker.reset()
        self.logger.info("Pipeline cleaned up.")
