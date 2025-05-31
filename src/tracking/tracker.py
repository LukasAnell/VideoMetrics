from typing import Optional, List

import cv2
import numpy as np

from ..detection.detector import Detection


class ObjectTracker:
    """Tracks objects across video frames using OpenCV trackers"""

    def __init__(self, tracker_type: str = "KCF"):
        """
        Initialize the tracker with specified algorithm

        Args:
            tracker_type: OpenCV tracker type ('KCF', 'CSRT', 'MOSSE', etc.)
        """
        self.tracker_type = tracker_type
        self.trackers = {} # id -> tracker
        self.tracks = {} # id -> (detection, age)
        self.next_id = 0

    def _create_tracker(self):
        """Create a tracker of the configured type"""
        if self.tracker_type == "KCF":
            return cv2.TrackerKCF.create()
        elif self.tracker_type == "CSRT":
            return cv2.TrackerCSRT.create()
        elif self.tracker_type == "MOSSE":
            return cv2.legacy.TrackerMOSSE.create()
        else:
            # Default to KCF if specified type isn't available
            return cv2.TrackerKCF_create()

    def update(self, frame: np.ndarray, detections: Optional[List[Detection]] = None) -> List[Detection]:
        """
        Update tracking with new frame and optional new detections

        Args:
            frame: Current video frame
            detections: New detections to incorporate (None if just updating existing tracks)

        Returns:
            List of tracked objects as Detection objects with track_id field
        """
        # Proces new detections if provided
        if detections:
            self._add_detections(frame, detections)

        # Update all trackers with new frame
        tracked_objects = []
        ids_to_remove = []

        for track_id, (tracker, detection, age) in self.tracks.items():
            success, box = tracker.update(frame)

            if success:
                # Update the detection with new position
                x1, y1, w, h = [int(v) for v in box]
                updated_detection = Detection(
                    bbox=[x1, y1, x1 + w, y1 + h],
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence
                )
                updated_detection.track_id = track_id
                tracked_objects.append(updated_detection)

                # Update the stored detection
                self.tracks[track_id] = (tracker, updated_detection, age + 1)
            else:
                # Mark for removal if tracking failed
                ids_to_remove.append(track_id)

        # Remove failed trackers
        for track_id in ids_to_remove:
            del self.tracks[track_id]

        return tracked_objects

    def _add_detections(self, frame: np.ndarray, detections: List[Detection]):
        """Add new detections as trackers"""
        for detection in detections:
            # Create new tracker
            tracker = self._create_tracker()

            # Initialize with detection bounding box
            x1, y1, x2, y2 = detection.bbox
            bbox = (x1, y1, x2 - x1, y2 - y1)
            tracker.init(frame, bbox)

            # Store with new ID
            track_id = self.next_id
            self.next_id += 1

            # Add track_id to detection object
            detection.track_id = track_id

            # Store tracker, detection, and age (0 = new)
            self.tracks[track_id] = (tracker, detection, 0)
