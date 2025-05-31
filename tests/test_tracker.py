import unittest
import numpy as np
import cv2
from src.detection.detector import Detection
from src.tracking.tracker import ObjectTracker


class TestObjectTracker(unittest.TestCase):
    def setUp (self):
        self.tracker = ObjectTracker("KCF")
        # Create a simple test frame
        self.frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw a rectangle to track
        cv2.rectangle(self.frame, (100, 100), (200, 200), (255, 255, 255), -1)

        # Create a test detection
        self.detection = Detection(
            bbox=(100, 100, 200, 200),
            class_id=1,
            class_name="test",
            confidence=0.9
        )

    def test_initialization (self):
        """Test tracker initialization"""
        self.assertEqual(self.tracker.tracker_type, "KCF")
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(self.tracker.next_id, 0)

    def test_add_detections (self):
        """Test adding detections to the tracker"""
        self.tracker._add_detections(self.frame, [self.detection])

        # Should have one track now
        self.assertEqual(len(self.tracker.tracks), 1)
        # Track ID should be assigned
        self.assertEqual(self.detection.track_id, 0)

        # Check the stored track
        tracker, detection, age = self.tracker.tracks[0]
        self.assertEqual(age, 0)  # New track

    def test_update_tracking (self):
        """Test updating tracks with a new frame"""
        # First add a detection
        self.tracker._add_detections(self.frame, [self.detection])

        # Create a slightly moved frame
        moved_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(moved_frame, (110, 110), (210, 210), (255, 255, 255), -1)

        # Update tracking
        tracked_objects = self.tracker.update(moved_frame)

        # Should still have one track
        self.assertEqual(len(self.tracker.tracks), 1)
        # Should return one tracked object
        self.assertEqual(len(tracked_objects), 1)

        # Age should be incremented
        _, _, age = self.tracker.tracks[0]
        self.assertEqual(age, 1)


if __name__ == '__main__':
    unittest.main()
