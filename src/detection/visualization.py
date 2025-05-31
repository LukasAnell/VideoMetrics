from typing import Optional, Tuple, List

import cv2
import numpy as np

from .detector import Detection


def draw_detection(frame: np.ndarray,
        detection: Detection,
        color: Optional[Tuple[int, int, int]] = None,
        thickness: int = 2,
        show_label: bool = True) -> np.ndarray:
    """
    Draw a single detection on a frame.

    Args:
        frame: Input frame
        detection: Detection object
        color: (B,G,R) color tuple (None for automatic color by class)
        thickness: Line thickness
        show_label: Whether to show class label and confidence

    Returns:
        Frame with detection visualization
    """
    if frame is None:
        return None

    result = frame.copy()

    # Extract bounding box
    x1, y1, x2, y2 = [int(v) for v in detection.bbox]

    # Generate color based on class_id if not provided
    if color is None:
        # Generate constistent color from class_id
        color_idx = detection.class_id % 20 # Limit to 20 distinct colors
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
            (64, 0, 64), (0, 64, 64), (192, 192, 192), (128, 128, 128)
        ]
        color = colors[color_idx]

    # Draw bounding box
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

    # Add label if requested
    if show_label:
        label_text = f"{detection.class_name}: {detection.confidence:.2f}"

        # Get size of text for background rectangle
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw filled rectangle
        cv2.rectangle(
            result,
            (x1, y1 - text_height - 4),
            (x1 + text_width, y1),
            color,
            cv2.FILLED
        )

        # Draw text
        cv2.putText(
            result,
            label_text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0), # Black text
            1,
            cv2.LINE_AA
        )

    return result

def draw_detections(frame: np.ndarray,
        detections: List[Detection],
        show_labels: bool = True) -> np.ndarray:
    """
    Draw multiple detections on a frame.

    Args:
        frame: Input frame
        detections: List of Detection objects
        show_labels: Whether to show class labels and confidence

    Returns:
        Frame with all detections visualized
    """
    if frame is None:
        return None

    result = frame.copy()

    for detection in detections:
        result = draw_detection(
            result,
            detection,
            show_label=show_labels
        )

    return result

def draw_stats(frame: np.ndarray,
        detections: List[Detection],
        position: Tuple[int, int] = (10, 30),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Draw detection statistics on frame.

    Args:
        frame: Input frame
        detections: List of detections
        position: Top-left position for stats display
        bg_color: Background color
        text_color: Text color

    Returns:
        Frame with stats overlay
    """
    if frame is None or not detections:
        return frame

    result = frame.copy()

    # Count objects by class
    class_counts = {}
    for det in detections:
        if det.class_name in class_counts:
            class_counts[det.class_name] += 1
        else:
            class_counts[det.class_name] = 1

    # Prepare text
    lines = [f"Total objects: {len(detections)}"]
    for cls_name, count in class_counts.items():
        lines.append(f"{cls_name}: {count}")

    # Draw stats box
    line_height = 20
    max_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in lines])

    x, y = position
    cv2.rectangle(
        result,
        (x, y - 15),
        (x + max_width + 10, y + line_height * len(lines)),
        bg_color,
        cv2.FILLED
    )

    # Draw text lines
    for i, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (x + 5, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA
        )

    return result
