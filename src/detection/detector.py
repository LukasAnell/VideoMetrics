import logging
import abc
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

class Detection:
    """Represents a single object detection result"""

    def __init__(self, bbox: Tuple[float, float, float, float],
            class_id: int,
            class_name: str,
            confidence: float,
            frame_id: Optional[int] = None):
        """
        Initialize a detection result.

        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2) in absolute pixels
            class_id: Class ID of the detected object
            class_name: Class name of the detected object
            confidence: Detection confidence score (0-1)
            frame_id: Optional frame identifier
        """
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.frame_id = frame_id
        self.track_id = None  # Optional tracking ID

    def __repr__(self):
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"

    @property
    def width(self) -> float:
        """Get width of the bounding box"""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Get height of bounding box"""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Get area of bounding box"""
        return self.width * self.height

class Detector(abc.ABC):
    """Base abstract class for all object detectors"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize a detection result.

        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2) in absolute pixels
            class_id: Class ID of the detected object
            class_name: Class name of the detected object
            confidence: Detection confidence score (0-1)
            frame_id: Optional frame identifier
        """
        self.config = config or {}
        self.model = None
        self.classes = []
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.initialized = False

    @abc.abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load detection model from path.

        Args:
            model_path: Path to model file or directory

        Returns:
            True if model loaded successfully
        """
        pass

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input frame for detection

        Returns:
            List of Detection objects
        """
        pass

    def filter_detections(self, detections: List[Detection],
            min_confidence: float = None,
            classes: List[str] = None) -> List[Detection]:
        """
        Filter detections by confidence and class.

        Args:
            detections: List of detections to filter
            min_confidence: Minimum confidence threshold (overrides instance default)
            classes: List of class names to keep (None = keep all)

        Returns:
            Filtered list of detections
        """
        threshold = min_confidence if min_confidence is not None else self.confidence_threshold

        filtered = [d for d in detections if d.confidence >= threshold]

        if classes:
            filtered = [d for d in filtered if d.class_name in classes]

        return filtered
