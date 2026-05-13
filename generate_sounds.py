"""
generate_sounds.py
===================
Tạo file âm thanh cảnh báo (.wav) bằng numpy — không cần tải về.
Chạy một lần trước khi dùng hệ thống:

    python generate_sounds.py
"""

import numpy as np
import wave
import struct
import os

SAMPLE_RATE = 44100
ASSETS_DIR  = "assets"


def make_tone(freq: float, duration: float, volume: float = 0.7,
              fade_ms: float = 10.0) -> np.ndarray:
    """Tạo sine wave với fade in/out để tránh click."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    wave_data = np.sin(2 * np.pi * freq * t)

    # Fade in/out
    fade_n = int(SAMPLE_RATE * fade_ms / 1000)
    envelope = np.ones(n)
    envelope[:fade_n]  = np.linspace(0, 1, fade_n)
    envelope[-fade_n:] = np.linspace(1, 0, fade_n)

    return (wave_data * envelope * volume * 32767).astype(np.int16)


def make_beep_sequence(freqs, duration_each=0.15, gap=0.08):
    """Ghép nhiều beep lại thành một sequence."""
    gap_samples = np.zeros(int(SAMPLE_RATE * gap), dtype=np.int16)
    parts = []
    for f in freqs:
        parts.append(make_tone(f, duration_each))
        parts.append(gap_samples)
    return np.concatenate(parts)


def save_wav(filename: str, data: np.ndarray):
    path = os.path.join(ASSETS_DIR, filename)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())
    print(f"  Saved: {path}")


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    print("Generating alert sounds...")

    # ── alert_normal.wav ──────────────────────────────────────────────────
    # 2 beep vừa phải — cảnh báo WARNING / DROWSY
    normal = make_beep_sequence([880, 1100], duration_each=0.18, gap=0.10)
    save_wav("alert_normal.wav", normal)

    # ── alert_urgent.wav ──────────────────────────────────────────────────
    # 4 beep nhanh và to hơn — cảnh báo CRITICAL
    urgent = make_beep_sequence(
        [1100, 1400, 1100, 1400],
        duration_each=0.12,
        gap=0.06,
    )
    # Volume cao hơn
    urgent = np.clip((urgent.astype(np.float32) * 1.3), -32767, 32767).astype(np.int16)
    save_wav("alert_urgent.wav", urgent)

    # ── calibration_done.wav ──────────────────────────────────────────────
    # Âm thanh dễ chịu báo hiệu calibration xong
    done = make_beep_sequence([440, 550, 660], duration_each=0.15, gap=0.05)
    save_wav("calibration_done.wav", done)

    print("Done! Files saved to assets/")
    print("  alert_normal.wav   — WARNING/DROWSY beep")
    print("  alert_urgent.wav   — CRITICAL beep (louder, faster)")
    print("  calibration_done.wav — calibration complete chime")


if __name__ == "__main__":
    main()
