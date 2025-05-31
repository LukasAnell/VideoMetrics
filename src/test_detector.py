import argparse
import logging
import os
import sys

import cv2

from detection.visualization import draw_detections, draw_stats
from detection.yolo_detector import YOLODetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main ():
    parser = argparse.ArgumentParser(description="Test object detection")
    parser.add_argument("--model", type=str, required=True, help="Path to model folder or weights file")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--output", type=str, help="Output directory for detected frames")
    args = parser.parse_args()

    # Create detector
    detector = YOLODetector(
        {
            "confidence_threshold": args.confidence
        }
    )

    # Load model
    if not detector.load_model(args.model):
        logger.error(f"Failed to load model: {args.model}")
        return 1

    # Open video source
    try:
        source = int(args.source)  # Try as camera index
    except ValueError:
        source = args.source  # Use as file path

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {args.source}")
        return 1

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detector.detect(frame)

        # Draw detections
        if detections:
            frame = draw_detections(frame, detections)
            frame = draw_stats(frame, detections)

        # Show result
        cv2.imshow("Detections", frame)

        # Save if requested
        if args.output and detections:
            os.makedirs(args.output, exist_ok=True)
            filename = os.path.join(args.output, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg")
            cv2.imwrite(filename, frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
