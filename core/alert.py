"""
core/alert.py
==============
Hệ thống cảnh báo âm thanh với logic leo thang (escalation).

Mức cảnh báo:
  WARNING  → beep nhẹ 1 lần
  DROWSY   → beep liên tục 2 lần
  CRITICAL → âm thanh khẩn cấp liên tục cho đến khi driver phản ứng

Chống spam: cooldown giữa các lần báo, reset khi driver tỉnh lại.
Dừng ngay: _stop_event được set khi driver tỉnh → thread âm thanh tự dừng.
"""

import time
import threading
import numpy as np
import logging
import os
from typing import Optional
from core.analyzer import DrowsinessLevel
import config


class AlertSystem:

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._last_alert_time = 0.0
        self._alert_count = 0
        self._current_level = DrowsinessLevel.AWAKE
        self._pygame_ok = False

        # Event để ra lệnh dừng giữa chừng cho thread âm thanh
        self._stop_event = threading.Event()

        self._init_pygame()

    def _init_pygame(self):
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._pygame_ok = True
            self.logger.info("pygame audio initialized")
        except ImportError:
            self.logger.warning("pygame not installed — audio alerts disabled")
        except Exception as e:
            self.logger.warning(f"pygame init failed: {e} — audio alerts disabled")

    def process_absence(self, absence_level):
        """Cảnh báo khi mất mặt liên tục — gọi từ pipeline."""
        from core.absence_handler import AbsenceLevel
        now = time.time()
        if now - self._last_alert_time < config.ALERT_COOLDOWN_SEC:
            return
        self._last_alert_time = now
        self._stop_event.clear()
        if absence_level == AbsenceLevel.CRITICAL:
            threading.Thread(target=self._play_sound,
                             args=(config.ALERT_SOUND_URGENT, 2), daemon=True).start()
        else:
            threading.Thread(target=self._play_sound,
                             args=(config.ALERT_SOUND_NORMAL, 1), daemon=True).start()

    def process(self, level: DrowsinessLevel):
        """
        Gọi mỗi frame với mức buồn ngủ hiện tại.
        Tự quản lý cooldown và escalation.
        """
        now = time.time()

        # Driver tỉnh lại → dừng âm thanh ngay lập tức và reset
        if level == DrowsinessLevel.AWAKE:
            if self._current_level != DrowsinessLevel.AWAKE:
                # Vừa chuyển từ buồn ngủ → tỉnh: ra lệnh dừng thread + mixer
                self._stop_event.set()
                self._stop_mixer()
            self._alert_count = 0
            self._current_level = DrowsinessLevel.AWAKE
            self._last_alert_time = 0.0  # Reset cooldown để báo ngay nếu ngủ lại
            return

        # Chưa đến lúc báo lại
        cooldown = config.ALERT_COOLDOWN_SEC
        if level == DrowsinessLevel.CRITICAL:
            cooldown = cooldown * 0.5   # báo nhanh hơn ở mức khẩn
        if now - self._last_alert_time < cooldown:
            return

        # Báo động — clear stop_event để thread mới chạy bình thường
        self._stop_event.clear()
        self._current_level = level
        self._last_alert_time = now
        self._alert_count += 1

        # Escalation: sau N lần không phản ứng → tăng mức
        if self._alert_count >= config.ALERT_ESCALATE_COUNT:
            level = DrowsinessLevel.CRITICAL

        threading.Thread(
            target=self._play_alert,
            args=(level,),
            daemon=True
        ).start()

    # ─────────────────────────────────────────────────────────────────────────

    def _stop_mixer(self):
        """Dừng tất cả channel mixer ngay lập tức."""
        if self._pygame_ok:
            try:
                import pygame
                pygame.mixer.stop()
            except Exception as e:
                self.logger.debug(f"mixer.stop failed: {e}")

    def _play_alert(self, level: DrowsinessLevel):
        if level == DrowsinessLevel.CRITICAL:
            self._play_sound(config.ALERT_SOUND_URGENT, repeat=3)
        elif level == DrowsinessLevel.DROWSY:
            self._play_sound(config.ALERT_SOUND_NORMAL, repeat=2)
        else:  # WARNING
            self._play_sound(config.ALERT_SOUND_NORMAL, repeat=1)

    def _play_sound(self, filepath: str, repeat: int = 1):
        # Thử dùng file âm thanh thật trước
        if self._pygame_ok and os.path.exists(filepath):
            try:
                import pygame
                sound = pygame.mixer.Sound(filepath)
                for _ in range(repeat):
                    if self._stop_event.is_set():
                        return   # ← driver đã tỉnh, dừng ngay
                    sound.play()
                    # Ngủ theo từng khoảng nhỏ để kiểm tra stop_event thường xuyên
                    end_time = time.time() + sound.get_length() + 0.1
                    while time.time() < end_time:
                        if self._stop_event.is_set():
                            pygame.mixer.stop()
                            return
                        time.sleep(0.05)
                return
            except Exception as e:
                self.logger.debug(f"Sound file play failed: {e}, using beep fallback")

        # Fallback: generate beep bằng numpy + pygame
        if self._pygame_ok:
            self._play_beep(repeat=repeat)
        else:
            # Terminal bell fallback cuối cùng
            for _ in range(repeat):
                if self._stop_event.is_set():
                    return
                print("\a", end="", flush=True)
                time.sleep(0.3)

    def _play_beep(self, freq: int = 880, duration_ms: int = 400, repeat: int = 1):
        """Generate và phát âm thanh beep thuần."""
        try:
            import pygame
            sample_rate = 44100
            t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000))
            wave = np.sin(2 * np.pi * freq * t)
            envelope = np.ones_like(wave)
            fade = int(sample_rate * 0.01)  # 10ms fade in/out
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
            wave = (wave * envelope * 32767).astype(np.int16)

            sound = pygame.sndarray.make_sound(wave)
            for _ in range(repeat):
                if self._stop_event.is_set():
                    return   # ← driver đã tỉnh, dừng ngay
                sound.play()
                # Chờ theo khoảng nhỏ để có thể hủy giữa chừng
                end_time = time.time() + duration_ms / 1000 + 0.1
                while time.time() < end_time:
                    if self._stop_event.is_set():
                        pygame.mixer.stop()
                        return
                    time.sleep(0.05)
        except Exception as e:
            self.logger.debug(f"Beep generation failed: {e}")

    def cleanup(self):
        self._stop_event.set()
        if self._pygame_ok:
            try:
                import pygame
                pygame.mixer.quit()
            except Exception:
                pass
