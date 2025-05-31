import unittest
import time
import numpy as np
from unittest.mock import MagicMock, patch
from src.metrics.collector import MetricsCollector
from src.detection.detector import Detection


class TestMetricsCollector(unittest.TestCase):
    def setUp (self):
        self.collector = MetricsCollector()

        # Create test frame and detections
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create sample detections with track IDs
        self.person_detection = Detection(
            bbox=(100, 100, 200, 200),
            class_id=0,
            class_name="person",
            confidence=0.9
        )
        self.person_detection.track_id = 1

        self.car_detection = Detection(
            bbox=(300, 300, 400, 400),
            class_id=2,
            class_name="car",
            confidence=0.85
        )
        self.car_detection.track_id = 2

    def test_initialization (self):
        """Test collector initialization"""
        self.assertEqual(self.collector.class_counts, {})
        self.assertEqual(self.collector.tracked_ids, set())
        self.assertEqual(self.collector.frame_count, 0)
        self.assertGreaterEqual(self.collector.start_time, time.time() - 1)

    def test_update_new_objects (self):
        """Test updating with new objects"""
        # Add two different objects
        self.collector.update([self.person_detection, self.car_detection], self.frame.shape)

        # Check class counts
        self.assertEqual(self.collector.class_counts, {"person": 1, "car": 1})

        # Check tracked IDs
        self.assertEqual(self.collector.tracked_ids, {1, 2})

        # Check frame count
        self.assertEqual(self.collector.frame_count, 1)

        # Check positions tracked
        self.assertEqual(len(self.collector.track_positions[1]), 1)
        self.assertEqual(len(self.collector.track_positions[2]), 1)

        # Verify centers calculated correctly
        person_center = self.collector.track_positions[1][0]
        self.assertEqual(person_center, (150, 150))  # Center of (100,100,200,200)

        car_center = self.collector.track_positions[2][0]
        self.assertEqual(car_center, (350, 350))  # Center of (300,300,400,400)

    def test_update_existing_objects (self):
        """Test updating existing objects doesn't increase count"""
        # First update
        self.collector.update([self.person_detection], self.frame.shape)

        # Move the person detection
        moved_person = Detection(
            bbox=(150, 150, 250, 250),
            class_id=0,
            class_name="person",
            confidence=0.9
        )
        moved_person.track_id = 1

        # Second update with same ID
        self.collector.update([moved_person], self.frame.shape)

        # Count should still be 1
        self.assertEqual(self.collector.class_counts, {"person": 1})

        # Should have 2 positions for trajectory
        self.assertEqual(len(self.collector.track_positions[1]), 2)

        # Check new position
        self.assertEqual(self.collector.track_positions[1][1], (200, 200))

    def test_get_summary (self):
        """Test summary statistics"""
        # Add objects
        self.collector.update([self.person_detection, self.car_detection], self.frame.shape)

        # Get summary
        summary = self.collector.get_summary()

        # Check summary contents
        self.assertEqual(summary["total_tracked"], 2)
        self.assertEqual(summary["class_counts"], {"person": 1, "car": 1})
        self.assertEqual(summary["current_in_frame"], 2)
        self.assertEqual(summary["processed_frames"], 1)

    def test_draw_metrics (self):
        """Test metrics visualization"""
        # Add objects
        self.collector.update([self.person_detection], self.frame.shape)

        # Add second position to create trajectory
        moved_person = Detection(
            bbox=(150, 150, 250, 250),
            class_id=0,
            class_name="person",
            confidence=0.9
        )
        moved_person.track_id = 1
        self.collector.update([moved_person], self.frame.shape)

        # Draw metrics
        result_frame = self.collector.draw_metrics(self.frame.copy())

        # Result should be a valid frame
        self.assertEqual(result_frame.shape, self.frame.shape)

        # Should have some non-zero pixels from drawing
        self.assertGreater(np.sum(result_frame), 0)


if __name__ == '__main__':
    unittest.main()
