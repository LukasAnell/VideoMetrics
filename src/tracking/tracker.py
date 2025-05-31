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
        self.tracks = {} # id -> (tracker, detection, age)
        self.next_id = 0
        self.max_age = 20 # Maximum frames to keep a track without updates
        self.iou_threshold = 0.5 # IoU threshold for matching

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
            return cv2.TrackerKCF.create()

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two boxes"""
        # Convert to [x1, y1, x2, y2] format if needed
        if len(box1) == 4:
            x1_1, y1_1, x2_1, y2_1 = box1
        else:
            x1_1, y1_1, w1, h1 = box1
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        if len(box2) == 4:
            x1_2, y1_2, x2_2, y2_2 = box2
        else:
            x1_2, y1_2, w_2, h_2 = box2
            x2_2, y2_2 = x1_2 + w_2, y1_2 + h_2

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0 # No intersection

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, frame: np.ndarray, detections: Optional[List[Detection]] = None) -> List[Detection]:
        """
        Update tracking with new frame and optional new detections

        Args:
            frame: Current video frame
            detections: New detections to incorporate (None if just updating existing tracks)

        Returns:
            List of tracked objects as Detection objects with track_id field
        """
        # Update all existing trackers with new frame
        tracked_objects = []
        ids_to_remove = []

        for track_id, (tracker, detection, age) in self.tracks.items():
            success, box = tracker.update(frame)

            if success:
                # Update the detection with new position
                x1, y1, w, h = [int(v) for v in box]
                updated_detection = Detection(
                    bbox=(x1, y1, x1 + w, y1 + h),
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

        # Remove failed or too old trackers
        for track_id in list(self.tracks.keys()):
            _, _, age = self.tracks[track_id]
            if track_id in ids_to_remove or age > self.max_age:
                del self.tracks[track_id]

        # Process new detections if provided
        if detections and len(detections) > 0:
            self._match_and_update(frame, detections, tracked_objects)

        return tracked_objects

    def _match_and_update(self, frame: np.ndarray, detections: List[Detection], tracked_objects: List[Detection]):
        """Match new detections with existing tracks and update accordingly"""
        if not tracked_objects: # No existing tracks, add all as new
            for detection in detections:
                self._add_detection(frame, detection)
            return

        # Calculate IoU between each detection and existing track
        matches = [] # list of (detection_idx, track_id, iou)

        for i, detection in enumerate(detections):
            for tracked in tracked_objects:
                iou = self._calculate_iou(detection.bbox, tracked.bbox)
                if iou > self.iou_threshold:
                    matches.append((i, tracked.track_id, iou))

        # Sort matches by IoU (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)

        # Track which detections and tracks have been matched
        matched_detections = set()
        matched_tracks = set()

        # Associate detections with tracks
        for det_idx, track_id, iou in matches:
            if det_idx not in matched_detections and track_id not in matched_tracks:
                # Update the existing track with new detection
                self._update_track(frame, track_id, detections[det_idx])
                matched_detections.add(det_idx)
                matched_tracks.add(track_id)

        # Add new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                self._add_detection(frame, detection)

    def _add_detection(self, frame: np.ndarray, detection: Detection):
        """Add a new detection as a tracker"""
        # Create a new tracker
        tracker = self._create_tracker()

        # Initialize with detection bounding box
        x1, y1, x2, y2 = detection.bbox
        bbox = (x1, y1, x2 - x1, y2 - y1)
        tracker.init(frame, bbox)

        # Store iwth new ID
        track_id = self.next_id
        self.next_id += 1

        # Add track_id to detection object
        detection.track_id = track_id

        # Store tracker, detection, and age (0 = new)
        self.tracks[track_id] = (tracker, detection, 0)

    def _update_track(self, frame: np.ndarray, track_id: int, detection: Detection):
        """Update an existing track with a new detection"""
        # Create a new tracker (re-initialize with the new detection)
        tracker = self._create_tracker()

        # Initialize with detection bounding box
        x1, y1, x2, y2 = detection.bbox
        bbox = (x1, y1, x2 - x1, y2 - y1)
        tracker.init(frame, bbox)

        # Keep the original track_id but update the detection
        detection.track_id = track_id

        # Reset age since we have a fresh detection
        self.tracks[track_id] = (tracker, detection, 0)

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
