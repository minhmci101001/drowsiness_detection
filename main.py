"""
Drowsiness Detection System
============================
Chạy file này để khởi động hệ thống.

Usage:
    python main.py
    python main.py --camera 0          # chọn camera index
    python main.py --no-yolo           # dùng MediaPipe face detect thay YOLO
    python main.py --demo              # chạy với video file thay camera thật
    python main.py --demo path/to/video.mp4
"""

import argparse
import sys
import cv2
from core.pipeline import DrowsinessPipeline
from utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Drowsiness Detection System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO face detection")
    parser.add_argument("--demo", nargs="?", const="demo", help="Run with video file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    logger.info("=== Drowsiness Detection System Starting ===")

    # Khởi tạo pipeline
    pipeline = DrowsinessPipeline(
        use_yolo=not args.no_yolo,
        logger=logger
    )

    # Chọn nguồn video
    if args.demo and args.demo != "demo":
        source = args.demo
        logger.info(f"Demo mode: {source}")
    elif args.demo == "demo":
        logger.info("Demo mode: using built-in test video")
        source = 0  # fallback to camera nếu không có file
    else:
        source = args.camera
        logger.info(f"Camera mode: index {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    logger.info("Press 'q' to quit | 'r' to recalibrate | 's' to save screenshot")

    try:
        pipeline.run(cap)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.cleanup()
        logger.info("System stopped.")

if __name__ == "__main__":
    main()
