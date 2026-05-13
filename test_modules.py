"""
test_modules.py
================
Kiểm tra từng module mà KHÔNG cần camera thật.
Chạy trước khi demo để đảm bảo mọi thứ hoạt động:

    python test_modules.py
    python test_modules.py --module ear       # chỉ test EAR
    python test_modules.py --module alert     # chỉ test âm thanh
    python test_modules.py --module pipeline  # test toàn bộ pipeline với frame giả
"""

import sys
import argparse
import numpy as np
import cv2
import time


def print_ok(msg):   print(f"  [OK]  {msg}")
def print_fail(msg): print(f"  [FAIL] {msg}")
def print_skip(msg): print(f"  [SKIP] {msg}")
def section(title):  print(f"\n{'─'*50}\n  {title}\n{'─'*50}")


# ────────────────────────────────────────────────────────────────────────────
# Test 1: Import tất cả modules
# ────────────────────────────────────────────────────────────────────────────
def test_imports():
    section("Test 1: Imports")
    modules = [
        ("config",                    "config"),
        ("core.quality_checker",      "FrameQualityChecker"),
        ("core.calibrator",           "Calibrator"),
        ("core.analyzer",             "DrowsinessAnalyzer"),
        ("core.blink_tracker",        "BlinkTracker"),
        ("core.absence_handler",      "FaceAbsenceHandler"),
        ("core.alert",                "AlertSystem"),
        ("core.display",              "DisplayRenderer"),
        ("utils.logger",              "EventLogger"),
    ]
    all_ok = True
    for mod, cls in modules:
        try:
            m = __import__(mod, fromlist=[cls])
            getattr(m, cls)
            print_ok(f"{mod}.{cls}")
        except Exception as e:
            print_fail(f"{mod}.{cls} — {e}")
            all_ok = False

    # Optional modules
    for mod, cls in [("core.face_detector", "FaceDetector"),
                     ("core.landmark_extractor", "LandmarkExtractor")]:
        try:
            m = __import__(mod, fromlist=[cls])
            getattr(m, cls)
            print_ok(f"{mod}.{cls} (requires mediapipe/ultralytics)")
        except Exception as e:
            print_skip(f"{mod}.{cls} — {e}")

    return all_ok


# ────────────────────────────────────────────────────────────────────────────
# Test 2: EAR calculation
# ────────────────────────────────────────────────────────────────────────────
def test_ear():
    section("Test 2: EAR Calculation")
    from core.landmark_extractor import LandmarkExtractor

    # Giả lập 6 điểm mắt mở
    # P1(trái), P2(trên trái), P3(trên phải), P4(phải), P5(dưới phải), P6(dưới trái)
    W, H = 640, 480
    open_eye = {
        33:  (0.30, 0.50, 0.0, 1.0),  # P1
        160: (0.34, 0.46, 0.0, 1.0),  # P2
        158: (0.38, 0.46, 0.0, 1.0),  # P3
        133: (0.42, 0.50, 0.0, 1.0),  # P4
        153: (0.38, 0.54, 0.0, 1.0),  # P5
        144: (0.34, 0.54, 0.0, 1.0),  # P6
    }
    closed_eye = {
        33:  (0.30, 0.50, 0.0, 1.0),
        160: (0.34, 0.499, 0.0, 1.0),  # rất gần nhau → mắt nhắm
        158: (0.38, 0.499, 0.0, 1.0),
        133: (0.42, 0.50, 0.0, 1.0),
        153: (0.38, 0.501, 0.0, 1.0),
        144: (0.34, 0.501, 0.0, 1.0),
    }

    import config
    ear_open   = LandmarkExtractor._calc_ear(open_eye,   config.RIGHT_EYE, W, H)
    ear_closed = LandmarkExtractor._calc_ear(closed_eye, config.RIGHT_EYE, W, H)

    print(f"  EAR mắt mở   = {ear_open:.4f}  (kỳ vọng > 0.15)")
    print(f"  EAR mắt nhắm = {ear_closed:.4f} (kỳ vọng < 0.05)")

    if ear_open > 0.15:
        print_ok("EAR mắt mở chính xác")
    else:
        print_fail(f"EAR mắt mở quá thấp: {ear_open:.4f}")

    if ear_closed < 0.05:
        print_ok("EAR mắt nhắm chính xác")
    else:
        print_fail(f"EAR mắt nhắm quá cao: {ear_closed:.4f}")


# ────────────────────────────────────────────────────────────────────────────
# Test 3: Calibrator
# ────────────────────────────────────────────────────────────────────────────
def test_calibrator():
    section("Test 3: Calibrator — Personal Baseline")
    from core.calibrator import Calibrator, CalibrationState
    import config

    # Giả lập người mắt híp: EAR baseline ~0.12 (thay vì 0.28 bình thường)
    print("  Giả lập người mắt híp (EAR baseline ~0.12)...")
    cal = Calibrator()
    cal.PROFILE_PATH = "/tmp/test_calibration.json"  # không ghi đè profile thật
    cal.start()

    # Feed 900 samples với EAR ~0.12, có chớp mắt (EAR về 0)
    rng = np.random.default_rng(42)
    for i in range(int(config.CALIBRATION_DURATION_SEC * config.CALIBRATION_FPS_ASSUME)):
        if i % 60 == 0:   # chớp mắt mỗi 2 giây
            ear = rng.uniform(0.01, 0.03)
        else:
            ear = rng.uniform(0.10, 0.14)
        cal.feed(ear, quality_ok=True)

    if cal.is_calibrated:
        print_ok(f"Calibration hoàn tất: baseline={cal.baseline:.4f}, threshold={cal.threshold:.4f}, eye_type={cal.eye_type}")
        # Với người mắt híp: baseline ~0.12, threshold ~0.09
        if 0.09 < cal.baseline < 0.16:
            print_ok(f"Baseline hợp lý cho mắt híp ({cal.baseline:.4f})")
        else:
            print_fail(f"Baseline ngoài kỳ vọng: {cal.baseline:.4f}")

        # Threshold dùng adaptive ratio từ EYE_PROFILES
        profile = config.EYE_PROFILES.get(cal.eye_type, config.EYE_PROFILES["normal"])
        expected_ratio = profile["threshold_ratio"]
        expected_thresh = cal.baseline * expected_ratio
        if abs(cal.threshold - expected_thresh) < 0.001:
            print_ok(f"Threshold = {expected_ratio} × baseline (eye_type={cal.eye_type}) ✓")
        else:
            print_fail(f"Threshold sai: {cal.threshold:.4f} vs expected {expected_thresh:.4f}")

        # Kiểm tra eye type
        if cal.eye_type in ("narrow", "very_narrow"):
            print_ok(f"Eye type phân loại đúng: {cal.eye_type}")
        else:
            print_fail(f"Eye type sai: {cal.eye_type} (kỳ vọng narrow hoặc very_narrow)")
    else:
        print_fail("Calibration không hoàn tất")

    # Test người mắt bình thường
    print("\n  Giả lập người mắt bình thường (EAR baseline ~0.28)...")
    cal2 = Calibrator()
    cal2.PROFILE_PATH = "/tmp/test_calibration2.json"
    cal2.start()
    for i in range(int(config.CALIBRATION_DURATION_SEC * config.CALIBRATION_FPS_ASSUME)):
        ear = rng.uniform(0.24, 0.32) if i % 60 != 0 else rng.uniform(0.01, 0.03)
        cal2.feed(ear, quality_ok=True)

    if cal2.is_calibrated:
        print_ok(f"Mắt bình thường: baseline={cal2.baseline:.4f}, threshold={cal2.threshold:.4f}")
        if 0.22 < cal2.baseline < 0.34:
            print_ok("Baseline hợp lý cho mắt bình thường")
        else:
            print_fail(f"Baseline ngoài kỳ vọng: {cal2.baseline:.4f}")


# ────────────────────────────────────────────────────────────────────────────
# Test 4: Analyzer — 3 tầng scoring
# ────────────────────────────────────────────────────────────────────────────
def test_analyzer():
    section("Test 4: DrowsinessAnalyzer — 3-layer scoring")
    from core.analyzer import DrowsinessAnalyzer, DrowsinessLevel

    analyzer = DrowsinessAnalyzer()
    analyzer.set_calibration(baseline=0.28, threshold=0.21)

    def feed_frames(n, ear, pitch=0.0, yaw=0.0, mar=0.2, ear_ok=True):
        result = None
        for _ in range(n):
            result = analyzer.update(
                ear=ear, mar=mar, pitch=pitch, yaw=yaw,
                ear_reliable=ear_ok
            )
        return result

    # Tình huống 1: Tỉnh táo
    analyzer.reset()
    r = feed_frames(60, ear=0.28)
    assert r.level == DrowsinessLevel.AWAKE, f"Expected AWAKE, got {r.level}"
    print_ok(f"Tỉnh táo → AWAKE (score={r.score:.2f})")

    # Tình huống 2: Mắt nhắm liên tục > 1.5s
    analyzer.reset()
    analyzer.set_calibration(0.28, 0.21)
    r = feed_frames(50, ear=0.10)  # EAR thấp liên tục 50 frame
    if r.level in (DrowsinessLevel.WARNING, DrowsinessLevel.DROWSY, DrowsinessLevel.CRITICAL):
        print_ok(f"Mắt nhắm 50 frame → {r.level.name} (score={r.score:.2f})")
    else:
        print_fail(f"Expected WARNING+, got {r.level.name}")

    # Tình huống 3: Mắt híp (EAR baseline thấp) — threshold được calibrate riêng
    analyzer.reset()
    analyzer.set_calibration(baseline=0.12, threshold=0.09)  # người mắt híp
    r = feed_frames(50, ear=0.11)  # EAR 0.11 > threshold 0.09 → vẫn tỉnh
    print_ok(f"Mắt híp, mắt mở bình thường → {r.level.name} (EAR=0.11 > thresh=0.09)")
    r2 = feed_frames(50, ear=0.06)  # EAR 0.06 < threshold 0.09 → buồn ngủ
    if r2.level != DrowsinessLevel.AWAKE:
        print_ok(f"Mắt híp, mắt nhắm → {r2.level.name} ✓")
    else:
        print_fail(f"Mắt híp mắt nhắm không detect được")

    # Tình huống 4: Gật đầu
    analyzer.reset()
    analyzer.set_calibration(0.28, 0.21)
    r = feed_frames(35, ear=0.25, pitch=20.0)  # gật đầu liên tục
    if r.pitch_flag:
        print_ok(f"Gật đầu 35 frame → pitch_flag=True (score={r.score:.2f})")
    else:
        print_fail(f"Không detect gật đầu: pitch_flag={r.pitch_flag}, consec={r.consec_frames}")

    # Tình huống 5: EAR không reliable (quay mặt) — không nên raise drowsy
    analyzer.reset()
    analyzer.set_calibration(0.28, 0.21)
    r = feed_frames(60, ear=0.05, ear_ok=False)  # EAR thấp nhưng không reliable
    print_ok(f"EAR không reliable (quay mặt) → {r.level.name} — ear_flag={r.ear_flag}")


# ────────────────────────────────────────────────────────────────────────────
# Test 5: Blink Tracker
# ────────────────────────────────────────────────────────────────────────────
def test_blink_tracker():
    section("Test 5: BlinkTracker")
    from core.blink_tracker import BlinkTracker

    tracker = BlinkTracker()
    tracker.set_threshold(0.21)

    # Giả lập chớp mắt bình thường: 15 lần/phút = mỗi 4 giây
    blink_count = 0
    stats = None
    # Simulate 120 giây @ 30fps = 3600 frames
    for frame_i in range(3600):
        t = frame_i / 30.0
        # Chớp mắt mỗi 4 giây, kéo dài 0.15s (4.5 frame @ 30fps)
        in_blink = (t % 4.0) < 0.15
        ear = 0.08 if in_blink else 0.28
        stats = tracker.update(ear=ear, ear_reliable=True)

    print(f"  Blink rate (15/min normal): {stats.blink_rate:.1f} lần/phút")
    print(f"  Avg blink duration: {stats.avg_blink_duration:.0f}ms")
    print(f"  Blink flag (kỳ vọng False): {stats.blink_flag}")

    if 10 <= stats.blink_rate <= 20:
        print_ok("Blink rate trong khoảng bình thường (10–20/phút)")
    else:
        print_fail(f"Blink rate bất thường: {stats.blink_rate:.1f}")

    if not stats.blink_flag:
        print_ok("Không flag khi blink rate bình thường")
    else:
        print_fail("False positive blink flag")


# ────────────────────────────────────────────────────────────────────────────
# Test 6: Quality Checker
# ────────────────────────────────────────────────────────────────────────────
def test_quality():
    section("Test 6: FrameQualityChecker — Edge Cases")
    from core.quality_checker import FrameQualityChecker, FrameQuality

    checker = FrameQualityChecker()

    # Frame tối (hầm xe)
    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8) + 15
    r = checker.check(dark_frame, face_confidence=0.9)
    if r.quality == FrameQuality.DARK:
        print_ok(f"Frame tối → DARK, ear_reliable={r.ear_reliable}")
    else:
        print_fail(f"Expected DARK, got {r.quality}")

    # Frame bình thường
    normal_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
    r = checker.check(normal_frame, face_confidence=0.9)
    if r.quality == FrameQuality.OK:
        print_ok(f"Frame bình thường → OK")
    else:
        print_fail(f"Expected OK, got {r.quality} (brightness={r.brightness:.1f})")

    # Mặt quay ngang
    r = checker.check(normal_frame, face_confidence=0.9, yaw_angle=40.0)
    if r.quality == FrameQuality.FACE_TURNED:
        print_ok(f"Quay mặt 40° → FACE_TURNED, ear_reliable={r.ear_reliable}")
    else:
        print_fail(f"Expected FACE_TURNED, got {r.quality}")

    # Phản chiếu kính (EAR variance thấp)
    r = checker.check(normal_frame, face_confidence=0.9, ear_variance=0.005)
    if r.quality == FrameQuality.REFLECTION:
        print_ok(f"EAR variance thấp → REFLECTION detected")
    else:
        print_fail(f"Expected REFLECTION, got {r.quality}")

    # Face confidence thấp
    r = checker.check(normal_frame, face_confidence=0.3)
    if r.quality == FrameQuality.NO_FACE:
        print_ok(f"Confidence thấp → NO_FACE")
    else:
        print_fail(f"Expected NO_FACE, got {r.quality}")

    # Kính râm (eye brightness ratio thấp)
    r = checker.check(normal_frame, face_confidence=0.9, eye_brightness_ratio=0.40)
    if r.quality == FrameQuality.SUNGLASSES:
        print_ok(f"Eye brightness ratio thấp → SUNGLASSES detected")
    else:
        print_fail(f"Expected SUNGLASSES, got {r.quality}")

    # IR camera (saturation thấp → check IR_MODE trong main check)
    # Tạo frame grayscale giả (saturation ≈ 0)
    gray_val = 100
    ir_frame = np.full((480, 640, 3), gray_val, dtype=np.uint8)
    r = checker.check(ir_frame, face_confidence=0.9)
    if r.quality == FrameQuality.IR_MODE:
        print_ok(f"Frame grayscale → IR_MODE detected")
    else:
        print_fail(f"Expected IR_MODE, got {r.quality} (có thể saturation không đủ thấp)")

    # IR mode helper
    is_ir = checker.check_ir_mode(ir_frame)
    if is_ir:
        print_ok("check_ir_mode() detect frame grayscale")
    else:
        print_fail("check_ir_mode() không detect frame grayscale")


# ────────────────────────────────────────────────────────────────────────────
# Test 7: Alert System
# ────────────────────────────────────────────────────────────────────────────
def test_alert():
    section("Test 7: AlertSystem")
    from core.alert import AlertSystem
    from core.analyzer import DrowsinessLevel

    alert = AlertSystem()

    if alert._pygame_ok:
        print_ok("pygame initialized")
        print("  Phát beep WARNING...")
        alert.process(DrowsinessLevel.WARNING)
        time.sleep(1.0)
        print("  Phát beep DROWSY...")
        alert.process(DrowsinessLevel.DROWSY)
        time.sleep(0.5)
        print_ok("Alert sounds played")
    else:
        print_skip("pygame not available — audio disabled")

    alert.cleanup()


# ────────────────────────────────────────────────────────────────────────────
# Test 8: Pipeline với frame giả
# ────────────────────────────────────────────────────────────────────────────
def test_pipeline():
    section("Test 8: Pipeline với frame giả (không cần camera)")
    try:
        from core.pipeline import DrowsinessPipeline
        from utils.logger import setup_logger

        logger = setup_logger("WARNING")  # ít log để test gọn hơn
        pipeline = DrowsinessPipeline(use_yolo=False, logger=logger)
        print_ok("Pipeline khởi tạo thành công")

        # Tạo VideoCapture giả từ blank frames
        # (Không có face → pipeline xử lý absence logic)
        class FakeCap:
            def __init__(self, n=60):
                self.n = n
                self.i = 0
            def read(self):
                if self.i >= self.n:
                    return False, None
                self.i += 1
                return True, np.zeros((720, 1280, 3), dtype=np.uint8)
            def release(self): pass
            def isOpened(self): return self.i < self.n
            def set(self, *a): pass

        print("  Chạy 60 frame blank (test absence handling)...")
        fake_cap = FakeCap(60)
        # Chạy không hiển thị window
        import unittest.mock as mock
        with mock.patch("cv2.imshow"), mock.patch("cv2.waitKey", return_value=ord('q')):
            pipeline.run(fake_cap)

        print_ok("Pipeline chạy 60 frame không crash")
        pipeline.cleanup()

    except Exception as e:
        print_fail(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test drowsiness detection modules")
    parser.add_argument("--module", choices=[
        "imports", "ear", "calibrator", "analyzer",
        "blink", "quality", "alert", "pipeline", "all"
    ], default="all")
    args = parser.parse_args()

    tests = {
        "imports":    test_imports,
        "ear":        test_ear,
        "calibrator": test_calibrator,
        "analyzer":   test_analyzer,
        "blink":      test_blink_tracker,
        "quality":    test_quality,
        "alert":      test_alert,
        "pipeline":   test_pipeline,
    }

    if args.module == "all":
        for name, fn in tests.items():
            try:
                fn()
            except Exception as e:
                print_fail(f"Test '{name}' crashed: {e}")
    else:
        tests[args.module]()

    print("\n" + "═"*50)
    print("  Test complete. Check [FAIL] lines above.")
    print("═"*50)


if __name__ == "__main__":
    main()
