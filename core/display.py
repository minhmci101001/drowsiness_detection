"""
core/display.py
================
Vẽ HUD (Heads-Up Display) lên frame video.

Hiển thị:
  - Trạng thái buồn ngủ (màu sắc thay đổi theo mức)
  - EAR, MAR realtime
  - PERCLOS bar
  - Landmark mắt và miệng
  - Head pose indicator
  - Cảnh báo quality (tối, quay mặt, phản chiếu)
  - Mini graph EAR theo thời gian
  - Trạng thái calibration
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple
from core.analyzer import DrowsinessLevel, AnalysisResult
from core.quality_checker import QualityReport, FrameQuality
from core.calibrator import Calibrator, CalibrationState
import config

# Màu sắc (BGR)
COLOR = {
    "awake":    (80,  200, 80 ),
    "warning":  (0,   200, 255),
    "drowsy":   (0,   120, 255),
    "critical": (0,   0,   255),
    "white":    (255, 255, 255),
    "gray":     (160, 160, 160),
    "black":    (0,   0,   0  ),
    "yellow":   (0,   220, 220),
    "cyan":     (220, 220, 0  ),
}

LEVEL_COLOR = {
    DrowsinessLevel.AWAKE:    COLOR["awake"],
    DrowsinessLevel.WARNING:  COLOR["warning"],
    DrowsinessLevel.DROWSY:   COLOR["drowsy"],
    DrowsinessLevel.CRITICAL: COLOR["critical"],
}

LEVEL_TEXT = {
    DrowsinessLevel.AWAKE:    "TINH TAO",
    DrowsinessLevel.WARNING:  "CHU Y",
    DrowsinessLevel.DROWSY:   "BUON NGU",
    DrowsinessLevel.CRITICAL: "!! NGUY HIEM !!",
}


class DisplayRenderer:

    EAR_HISTORY_LEN = 150   # 5 giây @ 30fps

    def __init__(self):
        self._ear_history = deque(maxlen=self.EAR_HISTORY_LEN)
        self._frame_count = 0

    def render(
        self,
        frame: np.ndarray,
        analysis: Optional[AnalysisResult],
        quality: Optional[QualityReport],
        calibrator: Calibrator,
        landmarks: Optional[Dict],
        fps: float,
        absence_report=None,   # AbsenceReport | None
    ) -> np.ndarray:
        """
        Vẽ toàn bộ HUD lên frame.
        Trả về frame đã được annotate (không modify in-place).
        """
        out = frame.copy()
        h, w = out.shape[:2]
        self._frame_count += 1

        # ── Calibration overlay (ưu tiên cao nhất) ───────────────────────
        if not calibrator.is_calibrated or calibrator.state == CalibrationState.COLLECTING:
            self._draw_calibration_overlay(out, calibrator, w, h)
            self._draw_fps(out, fps)
            return out

        # ── Quality warnings ──────────────────────────────────────────────
        if quality:
            self._draw_quality_warning(out, quality, w, h)

        # ── Absence warning ───────────────────────────────────────────────
        if absence_report and absence_report.is_absent:
            self._draw_absence_warning(out, absence_report, w, h)

        # ── Landmark overlay ──────────────────────────────────────────────
        if landmarks and config.SHOW_LANDMARKS:
            self._draw_landmarks(out, landmarks, w, h, analysis)

        if analysis is None:
            self._draw_fps(out, fps)
            return out

        # ── Lưu EAR history ───────────────────────────────────────────────
        if analysis.ear_reliable:
            self._ear_history.append(analysis.ear)

        # ── Status banner ─────────────────────────────────────────────────
        self._draw_status_banner(out, analysis, w, h)

        # ── Left panel: metrics ───────────────────────────────────────────
        self._draw_metrics_panel(out, analysis, w, h)

        # ── Right panel: mini graph ───────────────────────────────────────
        if config.SHOW_EAR_GRAPH:
            self._draw_ear_graph(out, analysis, w, h)

        # ── Score bar ─────────────────────────────────────────────────────
        self._draw_score_bar(out, analysis, w, h)

        # ── FPS ───────────────────────────────────────────────────────────
        self._draw_fps(out, fps)

        return out

    # ── Calibration ───────────────────────────────────────────────────────
    def _draw_calibration_overlay(self, frame, calibrator: Calibrator, w, h):
        if calibrator.state == CalibrationState.WAITING:
            msg1 = "CHUAN BI CALIBRATE..."
            msg2 = "Nhin thang vao camera, giu mat mo binh thuong"
            show_bar = False
        else:
            progress = calibrator.get_progress()
            pct = int(progress * 100)
            msg1 = f"DANG CALIBRATE... {pct}%"
            msg2 = "Giu mat mo binh thuong, khong di chuyen"
            show_bar = True

        # Overlay tối
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, msg1, (w//2 - 260, h//2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, COLOR["cyan"], 2)
        cv2.putText(frame, msg2, (w//2 - 300, h//2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR["white"], 1)
        cv2.putText(frame, "Nhan 'R' de calibrate lai bat cu luc nao",
                    (w//2 - 240, h//2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR["gray"], 1)

        if show_bar:
            progress = calibrator.get_progress()
            bar_w = int(w * 0.55)
            bar_x = (w - bar_w) // 2
            bar_y = h // 2 + 75
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16),
                          (60, 60, 60), -1)
            fill = int(bar_w * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 16),
                          COLOR["awake"], -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16),
                          COLOR["gray"], 1)

    # ── Status banner ─────────────────────────────────────────────────────
    def _draw_status_banner(self, frame, analysis: AnalysisResult, w, h):
        color = LEVEL_COLOR[analysis.level]
        text  = LEVEL_TEXT[analysis.level]

        # Flash khi critical
        if analysis.level == DrowsinessLevel.CRITICAL and (self._frame_count // 10) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), color, -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.rectangle(frame, (0, 0), (w, 65), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 65), color, 2)

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness  = 2
        (text_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - text_w) // 2
        cv2.putText(frame, text, (x, 45), font, font_scale, color, thickness)


    # ── Metrics panel ─────────────────────────────────────────────────────
    def _draw_metrics_panel(self, frame, analysis: AnalysisResult, w, h):
        panel_h = 250
        panel_w = 270
        panel_x, panel_y = 10, 75

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      COLOR["gray"], 1)

        def row(label, value, y_off, color=COLOR["white"], flag=False):
            y = panel_y + y_off
            flag_marker = " [!]" if flag else ""
            cv2.putText(frame, f"{label}: {value}{flag_marker}",
                        (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

        # EAR
        ear_color = COLOR["critical"] if analysis.ear_flag else COLOR["white"]
        row("EAR", f"{analysis.ear:.3f}", 28, ear_color, analysis.ear_flag)

        # Baseline + threshold + eye type (nhỏ hơn)
        if analysis.ear_baseline:
            eye_type_vn = {"normal": "BT", "narrow": "Hep", "very_narrow": "R.Hep"}
            et_label = eye_type_vn.get(analysis.eye_type, analysis.eye_type)
            cv2.putText(frame,
                        f"  base={analysis.ear_baseline:.3f}  thr={analysis.ear_threshold:.3f}  [{et_label}]",
                        (panel_x + 10, panel_y + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR["gray"], 1)

        # MAR / Yawn
        mar_color = COLOR["warning"] if analysis.yawn_flag else COLOR["white"]
        row("MAR", f"{analysis.mar:.3f}", 68, mar_color, analysis.yawn_flag)

        # PERCLOS
        perclos_color = COLOR["drowsy"] if analysis.perclos_flag else COLOR["white"]
        row("PERCLOS", f"{analysis.perclos*100:.1f}%", 91, perclos_color, analysis.perclos_flag)

        # Blink rate
        blink_color = COLOR["warning"] if analysis.blink_flag else COLOR["white"]
        blink_dur = f"  ({analysis.avg_blink_duration:.0f}ms)" if analysis.avg_blink_duration > 0 else ""
        row("Blink", f"{analysis.blink_rate:.0f}/min{blink_dur}", 114, blink_color, analysis.blink_flag)

        # Head pitch
        pitch_color = COLOR["warning"] if analysis.pitch_flag else COLOR["white"]
        row("Pitch", f"{analysis.pitch:.1f}°", 137, pitch_color, analysis.pitch_flag)

        # Yaw (chỉ hiện khi quay nhiều)
        if abs(analysis.yaw) > 15:
            yaw_color = COLOR["yellow"] if analysis.yaw_flag else COLOR["gray"]
            row("Yaw", f"{analysis.yaw:.1f}°", 160, yaw_color, analysis.yaw_flag)

        # EAR reliability warning
        if not analysis.ear_reliable:
            cv2.putText(frame, "! EAR KHONG DANG TIN",
                        (panel_x + 10, panel_y + 183),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR["yellow"], 1)

        # EAR Velocity (Cải tiến 2)
        if analysis.quick_drop_flag:
            cv2.putText(frame, f"!! NHAM MAT DOT NGOT (v={analysis.ear_velocity:.4f})",
                        (panel_x + 10, panel_y + 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR["critical"], 1)

        # Consec frames bar (nhỏ)
        if analysis.consec_frames > 0:
            # Dùng dynamic consec limit từ eye profile (Cải tiến 1)
            profile = config.EYE_PROFILES.get(analysis.eye_type, config.EYE_PROFILES["normal"])
            consec_limit = profile["ear_consec_frames"]
            bar_pct = min(analysis.consec_frames / consec_limit, 1.0)
            bw = int((panel_w - 20) * bar_pct)
            by = panel_y + 218
            cv2.rectangle(frame, (panel_x + 10, by), (panel_x + panel_w - 10, by + 8),
                          (60, 60, 60), -1)
            if bw > 0:
                bar_col = COLOR["critical"] if bar_pct >= 1.0 else COLOR["warning"]
                cv2.rectangle(frame, (panel_x + 10, by), (panel_x + 10 + bw, by + 8),
                              bar_col, -1)
            cv2.putText(frame, f"consec {analysis.consec_frames}f / {consec_limit}f",
                        (panel_x + 10, panel_y + 243),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR["gray"], 1)

    # ── EAR mini graph ────────────────────────────────────────────────────
    def _draw_ear_graph(self, frame, analysis: AnalysisResult, w, h):
        gw, gh = 250, 80
        gx, gy = w - gw - 10, 75

        overlay = frame.copy()
        cv2.rectangle(overlay, (gx, gy), (gx + gw, gy + gh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), COLOR["gray"], 1)
        cv2.putText(frame, "EAR history", (gx + 5, gy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR["gray"], 1)

        if len(self._ear_history) < 2:
            return

        # Vẽ threshold line
        if analysis.ear_threshold:
            thresh_y = int(gy + gh - (analysis.ear_threshold / 0.45) * (gh - 20) - 5)
            cv2.line(frame, (gx, thresh_y), (gx + gw, thresh_y), COLOR["drowsy"], 1)

        # Vẽ đường EAR
        pts = list(self._ear_history)
        n = len(pts)
        for i in range(1, n):
            x1 = int(gx + (i - 1) / max(self.EAR_HISTORY_LEN - 1, 1) * gw)
            x2 = int(gx + i       / max(self.EAR_HISTORY_LEN - 1, 1) * gw)
            y1 = int(gy + gh - (pts[i-1] / 0.45) * (gh - 20) - 5)
            y2 = int(gy + gh - (pts[i  ] / 0.45) * (gh - 20) - 5)
            y1 = max(gy + 18, min(gy + gh - 2, y1))
            y2 = max(gy + 18, min(gy + gh - 2, y2))
            cv2.line(frame, (x1, y1), (x2, y2), COLOR["awake"], 1)

    # ── Score bar ─────────────────────────────────────────────────────────
    def _draw_score_bar(self, frame, analysis: AnalysisResult, w, h):
        bar_y = h - 30
        bar_w = int(w * 0.5)
        bar_x = (w - bar_w) // 2

        cv2.putText(frame, "DROWSY SCORE",
                    (bar_x - 130, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR["gray"], 1)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (40, 40, 40), -1)

        fill  = int(bar_w * min(analysis.score, 1.0))
        color = LEVEL_COLOR[analysis.level]
        if fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 18), color, -1)

        # Threshold lines
        for thresh, label in [
            (config.SCORE_ALERT_THRESHOLD,  ""),
            (config.SCORE_URGENT_THRESHOLD, ""),
        ]:
            tx = bar_x + int(bar_w * thresh)
            cv2.line(frame, (tx, bar_y - 4), (tx, bar_y + 22), COLOR["white"], 1)

        cv2.putText(frame, f"{analysis.score:.2f}",
                    (bar_x + bar_w + 8, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ── Landmarks ─────────────────────────────────────────────────────────
    def _draw_landmarks(self, frame, landmarks: Dict, w, h, analysis: Optional[AnalysisResult]):
        def draw_eye(indices, color):
            pts = []
            for i in indices:
                if i in landmarks:
                    lm = landmarks[i]
                    pts.append((int(lm[0] * w), int(lm[1] * h)))
            if len(pts) >= 3:
                pts_np = np.array(pts, dtype=np.int32)
                cv2.polylines(frame, [pts_np], isClosed=True, color=color, thickness=1)
                for pt in pts:
                    cv2.circle(frame, pt, 2, color, -1)

        eye_color = COLOR["critical"] if (analysis and analysis.ear_flag) else COLOR["cyan"]
        draw_eye(config.LEFT_EYE,  eye_color)
        draw_eye(config.RIGHT_EYE, eye_color)

        mouth_color = COLOR["warning"] if (analysis and analysis.yawn_flag) else COLOR["gray"]
        draw_eye(config.MOUTH, mouth_color)

    # ── Absence warning ───────────────────────────────────────────────────
    def _draw_absence_warning(self, frame, absence_report, w, h):
        from core.absence_handler import AbsenceLevel
        dur = absence_report.duration_sec

        color_map = {
            AbsenceLevel.SHORT:    COLOR["gray"],
            AbsenceLevel.WARNING:  COLOR["warning"],
            AbsenceLevel.ALERT:    COLOR["drowsy"],
            AbsenceLevel.CRITICAL: COLOR["critical"],
        }
        color = color_map.get(absence_report.level, COLOR["gray"])

        msg = f"KHONG THAY MAT — {dur:.1f}s"
        if absence_report.level == AbsenceLevel.CRITICAL:
            msg = f"!! MAT MAT QUA LAU — {dur:.1f}s !!"

        # Flash khi critical
        if absence_report.level == AbsenceLevel.CRITICAL and (self._frame_count // 8) % 2 == 0:
            return  # blink effect — bỏ qua frame chẵn

        cv2.putText(frame, msg,
                    (w // 2 - len(msg) * 7, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # ── Quality warning ───────────────────────────────────────────────────
    def _draw_quality_warning(self, frame, quality: QualityReport, w, h):
        yaw_str = f"{quality.yaw_angle:.0f}" if quality.yaw_angle is not None else "?"
        msgs = {
            FrameQuality.DARK:        "ANH SANG THAP — EAR KHONG CHINH XAC",
            FrameQuality.OVEREXPOSED: "ANH SANG QUA MANH",
            FrameQuality.FACE_TURNED: f"MAT QUAY NGANG ({yaw_str} deg) — BO QUA EAR",
            FrameQuality.REFLECTION:  "PHAN CHIEU KINH — EAR KHONG DANG TIN",
            FrameQuality.BLUR:        "FRAME BI MO",
            FrameQuality.LOW_CONF:    "LANDMARK THAP — CO THE BI CHE MAT",
            FrameQuality.NO_FACE:     "KHONG THAY MAT — KIEM TRA CAMERA",
            FrameQuality.SUNGLASSES:  "KINH RAM DETECTED — BO QUA EAR, DUNG HEAD POSE",
            FrameQuality.IR_MODE:     "CHE DO IR (BAN DEM) — DO CHINH XAC GIAM",
        }
        msg = msgs.get(quality.quality)
        if msg:
            cv2.putText(frame, msg, (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR["yellow"], 1)

    # ── Absence warning ───────────────────────────────────────────────────
    def draw_absence_warning(self, frame, duration_sec: float, level_str: str):
        """Gọi từ pipeline khi mất mặt liên tục."""
        h, w = frame.shape[:2]
        msg = f"MAT MAT {duration_sec:.1f}s — {level_str.upper()}"
        color = COLOR["critical"] if duration_sec > 5 else COLOR["warning"]
        if (self._frame_count // 8) % 2 == 0:
            cv2.putText(frame, msg, (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # ── FPS ───────────────────────────────────────────────────────────────
    def _draw_fps(self, frame, fps: float):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR["gray"], 1)
