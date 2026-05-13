# Drowsiness Detection System

Hệ thống phát hiện buồn ngủ khi lái xe sử dụng Computer Vision.  
Hỗ trợ adaptive cho mắt híp, kính râm, khẩu trang, camera IR ban đêm.

## Kiến trúc pipeline

```
Camera frame
    ↓
YOLO (face detection) → Haar Cascade fallback nếu YOLO không có
  └── Frame Skipping: chỉ chạy YOLO mỗi 5 frame (tiết kiệm GPU)
    ↓
MediaPipe FaceLandmarker (VIDEO mode — 478 landmarks + tracking)
    ↓
FrameQualityChecker (lọc frame xấu)
    ├── Tối / chói / mờ → skip EAR
    ├── Quay mặt > 35°  → skip EAR
    ├── Phản chiếu kính → skip EAR
    ├── Kính râm         → skip EAR, dùng Head Pose
    └── Camera IR        → giảm tin cậy EAR
    ↓
Calibrator (5s đầu — auto-start khi mặt ổn định)
    ├── Baseline = median EAR lúc tỉnh (loại bỏ chớp mắt)
    └── Phân loại eye type: normal / narrow / very_narrow
    ↓
┌──────────────────────────────────────────────────┐
│ DrowsinessAnalyzer — 3 tầng (adaptive weights)   │
│  Tầng 1: EAR < adaptive_ratio × baseline        │
│  Tầng 2: PERCLOS > 15% trong 60s                │
│  Tầng 3: Weighted score (thay đổi theo eye type)│
│    + EAR Velocity tracking (microsleep)          │
└──────────────────────────────────────────────────┘
    ↓
BlinkTracker (blink rate + slow blink)
    ↓
FaceAbsenceHandler (mất mặt liên tục)
    ↓
AlertSystem (beep + escalation)
    ↓
DisplayRenderer (HUD + EAR graph + absence overlay)
    ↓
EventLogger (CSV log mọi sự kiện)
```

## Cài đặt

```bash
pip install -r requirements.txt
python generate_sounds.py   # tạo file âm thanh (chạy 1 lần)
python test_modules.py      # kiểm tra không cần camera
python main.py              # chạy hệ thống
```

## Tùy chọn khi chạy

```bash
python main.py                      # camera mặc định (index 0)
python main.py --camera 1           # chọn camera khác
python main.py --no-yolo            # chỉ dùng Haar Cascade face detection
python main.py --demo video.mp4     # chạy với video file
python main.py --log-level DEBUG    # log chi tiết
```

## Phím tắt

| Phím | Chức năng |
|------|-----------|
| `C`  | Force bắt đầu calibrate thủ công |
| `R`  | Recalibrate lại từ đầu (reset toàn bộ state) |
| `S`  | Chụp screenshot |
| `D`  | Bật/tắt landmark + debug overlay |
| `Q`  | Thoát |

## Các trường hợp đặc biệt được xử lý

| Tình huống | Module xử lý | Cách xử lý |
|-----------|-------------|-----------|
| **Mắt híp (Adaptive)** | Calibrator + Analyzer | Phân loại `normal`/`narrow`/`very_narrow`, dùng adaptive threshold ratio (0.75/0.70/0.65) và dynamic scoring weights |
| **Microsleep (EAR Velocity)** | Analyzer | Theo dõi tốc độ giảm EAR; nếu sụt đột ngột (`< -0.025/frame`) → `quick_drop_flag` cảnh báo ngay |
| **Kính râm** | QualityChecker | So sánh brightness vùng mắt vs mặt; ratio < 0.55 → skip EAR, dùng Head Pose |
| **Khẩu trang** | Pipeline | Kiểm tra `mouth_visibility`; nếu < 0.4 → bỏ qua MAR scoring (không tính ngáp) |
| **Camera IR ban đêm** | QualityChecker | Saturation thấp (< 20) → flag IR_MODE, giảm tin cậy EAR |
| **Đeo kính (phản chiếu)** | QualityChecker | EAR variance thấp bất thường → skip EAR frame |
| **Ánh sáng tối** | QualityChecker | HSV brightness < 40 → flag DARK, skip EAR |
| **Ánh sáng chói** | QualityChecker | Brightness > 220 → flag OVEREXPOSED |
| **Quay mặt ngang** | QualityChecker | Head yaw > 35° → skip EAR |
| **Gật đầu** | Analyzer | Pitch > 15° liên tiếp 1s → pitch_flag |
| **Ngáp** | Analyzer | MAR > 0.6 liên tục 20 frame → yawn_flag |
| **Blink rate thấp** | BlinkTracker | < 10 lần/phút → blink_flag |
| **Slow blink** | BlinkTracker | Nhắm > 400ms/lần → slow blink warning |
| **Mất mặt liên tục** | AbsenceHandler | > 2s warning, > 5s alert, > 10s critical |
| **Frame bị mờ** | QualityChecker | Laplacian variance < 50 → skip |
| **Xe rung (EAR spike)** | Analyzer | Exponential moving average làm mượt EAR |

## Tối ưu hiệu năng

| Tối ưu | Chi tiết |
|--------|---------|
| **MediaPipe VIDEO mode** | Dùng `detect_for_video()` với timestamp — kích hoạt internal tracking giữa các frame, giảm jitter |
| **YOLO Frame Skipping** | Chỉ chạy YOLO mỗi 5 frame (`FACE_DETECTION_SKIP`). Khi mất mặt, detect mỗi frame để recovery nhanh |
| **Smart skip logic** | Bypass frame skip khi `_last_face = None` — đảm bảo absence handler trigger đúng lúc |

## Cấu trúc file

```
drowsiness_detection/
├── main.py                    # Entry point
├── config.py                  # Tất cả thông số — chỉnh ở đây
├── requirements.txt
├── generate_sounds.py         # Tạo file .wav cảnh báo (chạy 1 lần)
├── test_modules.py            # Test 8 modules không cần camera
├── core/
│   ├── face_detector.py       # YOLO + Haar Cascade face detection
│   ├── landmark_extractor.py  # 478 landmarks + EAR/MAR/head pose
│   ├── quality_checker.py     # Lọc frame xấu (tối, quay, kính râm, IR...)
│   ├── calibrator.py          # Baseline cá nhân + eye type classification
│   ├── analyzer.py            # 3-layer drowsiness scoring (adaptive)
│   ├── blink_tracker.py       # Blink rate + slow blink detection
│   ├── absence_handler.py     # Xử lý mất mặt liên tục
│   ├── alert.py               # Âm thanh cảnh báo + escalation
│   ├── display.py             # HUD + EAR graph + absence overlay
│   └── pipeline.py            # Kết nối tất cả, vòng lặp chính
└── utils/
    └── logger.py              # Logging + CSV event log
```

## Tuning thông số (config.py)

| Thông số | Mặc định | Ý nghĩa |
|---------|---------|---------|
| `CALIBRATION_DURATION_SEC` | 5 | Thời gian calibration (giây) |
| `CALIBRATION_THRESHOLD_RATIO` | 0.75 | Ratio mặc định (bị override bởi EYE_PROFILES) |
| `EYE_TYPE_NORMAL_MIN` | 0.24 | EAR baseline ≥ 0.24 → "normal" |
| `EYE_TYPE_NARROW_MIN` | 0.16 | EAR baseline ≥ 0.16 → "narrow", < 0.16 → "very_narrow" |
| `EAR_CONSEC_FRAMES` | 45 | Số frame EAR thấp liên tục (1.5s @ 30fps) |
| `EAR_VELOCITY_DROP_THRESH` | -0.025 | Tốc độ giảm EAR/frame → quick drop flag |
| `FACE_DETECTION_SKIP` | 5 | YOLO chạy mỗi N frame |
| `PERCLOS_ALERT_THRESHOLD` | 0.15 | 15% mắt nhắm trong 60s → alert |
| `SCORE_ALERT_THRESHOLD` | 0.50 | Score > 0.5 → DROWSY |
| `SCORE_URGENT_THRESHOLD` | 0.80 | Score > 0.8 → CRITICAL |
| `HEAD_PITCH_THRESHOLD` | 15.0° | Góc gật đầu tính là buồn ngủ |
| `BLINK_RATE_LOW_THRESHOLD` | 10/min | Blink rate thấp hơn → flag |
| `SUNGLASSES_BRIGHTNESS_RATIO` | 0.55 | Eye/face brightness < 0.55 → kính râm |
| `IR_SATURATION_THRESHOLD` | 20 | Saturation < 20 → IR mode |
| `MOUTH_VISIBILITY_MIN` | 0.4 | Visibility < 0.4 → đeo khẩu trang, bỏ MAR |
