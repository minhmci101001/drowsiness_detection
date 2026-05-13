"""
config.py — Tất cả thông số của hệ thống tập trung tại đây.
Chỉnh sửa file này để tune hệ thống mà không cần đụng vào logic.
"""

# ─────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────
CALIBRATION_DURATION_SEC = 5       # thời gian calibrate lúc đầu (giây)
CALIBRATION_FPS_ASSUME   = 30      # fps giả định khi tính số frame
CALIBRATION_THRESHOLD_RATIO = 0.75 # threshold = ratio × EAR_baseline

# ─────────────────────────────────────────────
# EAR (Eye Aspect Ratio)
# ─────────────────────────────────────────────
EAR_CONSEC_FRAMES   = 50           # số frame liên tiếp EAR thấp → flag (~5s @ 10fps)
EAR_HARD_MIN        = 0.10         # EAR tuyệt đối thấp hơn này = nhắm mắt dù baseline thế nào
EAR_HARD_MAX        = 0.40         # EAR cao hơn này = mắt mở hoàn toàn
EAR_SMOOTH_ALPHA    = 0.3          # hệ số làm mượt EAR (exponential moving average)

# ─────────────────────────────────────────────
# PERCLOS (Percentage Eye Closure)
# ─────────────────────────────────────────────
PERCLOS_WINDOW_SEC   = 60          # cửa sổ thời gian tính PERCLOS (giây)
PERCLOS_ALERT_THRESHOLD = 0.15     # 15% → cảnh báo
PERCLOS_URGENT_THRESHOLD = 0.30    # 30% → khẩn cấp

# ─────────────────────────────────────────────
# MAR (Mouth Aspect Ratio) — phát hiện ngáp
# ─────────────────────────────────────────────
MAR_YAWN_THRESHOLD  = 0.60         # MAR cao hơn này = ngáp
MAR_YAWN_CONSEC     = 20           # số frame liên tiếp → xác nhận ngáp
MAR_SMOOTH_ALPHA    = 0.4

# ─────────────────────────────────────────────
# HEAD POSE
# ─────────────────────────────────────────────
HEAD_PITCH_THRESHOLD = 15.0        # độ gật đầu xuống (degree)
HEAD_ROLL_THRESHOLD  = 20.0        # độ nghiêng đầu (degree)
HEAD_YAW_THRESHOLD   = 35.0        # độ quay mặt ngang — quá ngưỡng này skip EAR
HEAD_POSE_CONSEC     = 30          # số frame liên tiếp → flag

# ─────────────────────────────────────────────
# SCORING (tầng 3)
# ─────────────────────────────────────────────
SCORE_WEIGHT_EAR        = 0.45
SCORE_WEIGHT_HEAD_POSE  = 0.20
SCORE_WEIGHT_YAWN       = 0.15
SCORE_WEIGHT_PERCLOS    = 0.10
SCORE_WEIGHT_BLINK      = 0.10   # blink rate thấp

SCORE_ALERT_THRESHOLD   = 0.50     # score > 0.5 → cảnh báo nhẹ
SCORE_URGENT_THRESHOLD  = 0.80     # score > 0.8 → cảnh báo khẩn

# ─────────────────────────────────────────────
# QUALITY CHECKS — frame reliability
# ─────────────────────────────────────────────
FACE_DETECTION_SKIP     = 5        # Chạy YOLO mỗi n frame
MIN_FACE_CONFIDENCE     = 0.75     # YOLO/MediaPipe confidence tối thiểu
MIN_LANDMARK_VISIBILITY = 0.6      # visibility của landmark tối thiểu
MIN_BRIGHTNESS          = 40       # pixel brightness trung bình tối thiểu (0–255)
MAX_BRIGHTNESS          = 220      # quá sáng cũng bỏ qua (phản chiếu kính)
YAW_SKIP_THRESHOLD      = 35.0     # quay mặt quá ngưỡng này → bỏ qua EAR frame đó
REFLECTION_EAR_VAR_MAX  = 0.02     # EAR variance thấp bất thường → nghi phản chiếu kính

# ─────────────────────────────────────────────
# BLINK RATE
# ─────────────────────────────────────────────
BLINK_RATE_LOW_THRESHOLD  = 10.0   # < 10 lần/phút → flag buồn ngủ
BLINK_SLOW_DURATION_MS    = 400    # nhắm > 400ms = slow blink
BLINK_WINDOW_SEC          = 60     # cửa sổ đếm blink

# ─────────────────────────────────────────────
# ALERT
# ─────────────────────────────────────────────
ALERT_COOLDOWN_SEC      = 5        # không báo liên tục, chờ ít nhất N giây
ALERT_ESCALATE_COUNT    = 3        # sau N lần báo mà không phản ứng → escalate
ALERT_SOUND_NORMAL      = "assets/alert_normal.wav"
ALERT_SOUND_URGENT      = "assets/alert_urgent.wav"
ALERT_SOUND_CALIB_DONE  = "assets/calibration_done.wav"

# ─────────────────────────────────────────────
# EDGE CASES ĐẶC BIỆT
# ─────────────────────────────────────────────
# Mắt nhắm một bên (microsleep bất đối xứng)
EAR_ASYMMETRY_THRESHOLD = 0.10   # |EAR_left - EAR_right| > này → flag bất đối xứng

# Khẩu trang / che mặt dưới — landmark visibility miệng thấp
MOUTH_VISIBILITY_MIN    = 0.4    # dưới này → bỏ qua MAR

# Mũ che trán — landmark forehead visibility thấp → head pose kém tin
FOREHEAD_LANDMARK_IDX   = [10, 151, 9, 8]   # indices landmark trán
FOREHEAD_VISIBILITY_MIN = 0.5

# Microsleep: nhắm mắt cực ngắn nhưng đột ngột (khác chớp mắt bình thường)
# Chớp mắt bình thường: ~150–400ms. Microsleep: >500ms
MICROSLEEP_FRAMES       = 15    # @ 30fps = 0.5s — ngắn hơn EAR_CONSEC nhưng vẫn đáng lo

# Tần suất chớp mắt (blink rate) — bổ sung
BLINK_RATE_HIGH_THRESH  = 25    # > 25 lần/phút = mắt mỏi / kích ứng

# ─────────────────────────────────────────────
# ADAPTIVE EYE TYPE — Cải tiến 1: Mắt híp
# ─────────────────────────────────────────────
# Phân loại dựa trên EAR baseline sau calibration
EYE_TYPE_NORMAL_MIN     = 0.24   # baseline >= 0.24 → "normal"
EYE_TYPE_NARROW_MIN     = 0.16   # 0.16 <= baseline < 0.24 → "narrow"
# baseline < 0.16 → "very_narrow"

# Profile cho từng loại mắt:
#   threshold_ratio   — tỉ lệ threshold/baseline (thấp hơn = ít nhạy hơn)
#   ear_consec_frames — số frame EAR thấp liên tiếp cần thiết
#   score_weight_*    — trọng số scoring (tổng = 1.0)
EYE_PROFILES = {
    "normal": {
        "threshold_ratio":      0.80,
        "ear_consec_frames":    50,    # ~5s tại 10fps thực tế
        "score_weight_ear":     0.45,
        "score_weight_head_pose": 0.20,
        "score_weight_yawn":    0.15,
        "score_weight_perclos": 0.10,
        "score_weight_blink":   0.10,
    },
    "narrow": {
        "threshold_ratio":      0.80,
        "ear_consec_frames":    50,    # ~5s tại 10fps thực tế
        "score_weight_ear":     0.45,
        "score_weight_head_pose": 0.20,
        "score_weight_yawn":    0.15,
        "score_weight_perclos": 0.10,
        "score_weight_blink":   0.10,
    },
    "very_narrow": {
        "threshold_ratio":      0.78,
        "ear_consec_frames":    60,    # ~6s tại 10fps
        "score_weight_ear":     0.35,
        "score_weight_head_pose": 0.25,
        "score_weight_yawn":    0.15,
        "score_weight_perclos": 0.15,
        "score_weight_blink":   0.10,
    },
}

# ─────────────────────────────────────────────
# EAR VELOCITY — Cải tiến 2
# ─────────────────────────────────────────────
# Theo dõi tốc độ thay đổi EAR (phát hiện nhắm mắt đột ngột)
EAR_VELOCITY_WINDOW     = 5      # số frame dùng tính velocity
EAR_VELOCITY_DROP_THRESH = -0.025  # EAR giảm trung bình/frame → quick drop
SCORE_WEIGHT_VELOCITY   = 0.10   # weight bổ sung khi có quick drop

# ─────────────────────────────────────────────
# SUNGLASSES DETECTION — Cải tiến 3a
# ─────────────────────────────────────────────
# So sánh brightness vùng mắt vs vùng mặt
SUNGLASSES_BRIGHTNESS_RATIO = 0.55  # eye_brightness < 55% face_brightness → kính râm

# ─────────────────────────────────────────────
# IR CAMERA DETECTION — Cải tiến 3c
# ─────────────────────────────────────────────
# Camera hồng ngoại ban đêm → frame gần như grayscale
IR_SATURATION_THRESHOLD = 20    # mean(S in HSV) < 20 → IR mode
IR_FACE_CONFIDENCE_BOOST = 0.10  # giảm min face confidence trong IR mode

# ─────────────────────────────────────────────
# MEDIAPIPE LANDMARK INDICES
# ─────────────────────────────────────────────
# Mắt trái (từ góc nhìn người nhìn vào camera)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Mắt phải
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
# Miệng (8 điểm)
MOUTH     = [61, 39, 0, 269, 291, 405, 17, 181]
# Điểm dùng để tính head pose (nose tip, chin, eye corners, mouth corners)
HEAD_POSE_POINTS = [1, 152, 33, 263, 61, 291]

# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────
DISPLAY_WIDTH   = 1280
DISPLAY_HEIGHT  = 720
SHOW_LANDMARKS  = True
SHOW_EAR_GRAPH  = True
SHOW_DEBUG_INFO = True
