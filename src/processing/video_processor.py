import logging
import time
from typing import Dict, Tuple, List

import numpy as np

from src.detection.detector import Detector, Detection
from src.tracking.tracker import ObjectTracker

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Processes video frames with detection and tracking"""

    def __init__(self, detector: Detector, config: Dict = None):
        """
        Initialize with detector and configuration

        Args:
            detector: Object detector implementation
            config: Configuration dictionary
        """
        self.detector = detector
        self.config = config or {}

        # Initialize tracker
        tracker_type = self.config.get("tracker_type", "KCF")
        self.tracker = ObjectTracker(tracker_type)

        # Detection frequency (every N frames)
        self.detection_interval = self.config.get("detection_interval", 5)
        self.frame_count = 0

        # Performance metrics
        self.fps = 0
        self.processing_time = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        """
        Process a single frame with detection and tracking

        Args:
            frame: Input video frame

        Returns:
            Tuple of (processed frame, detections)
        """
        start_time = time.time()

        # Run detection on selected frames
        detections = None
        if self.frame_count % self.detection_interval == 0:
            detections = self.detector.detect(frame)

        # Update tracking with new frame (and detections if available)
        tracked_objects = self.tracker.update(frame, detections)

        # Update performance metrics
        self.processing_time = time.time() - start_time
        self.fps = 1.0 / self.processing_time if self.processing_time > 0 else 0

        # Increment frame counter
        self.frame_count += 1

        return frame, tracked_objects
