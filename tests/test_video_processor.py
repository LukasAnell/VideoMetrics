import unittest
import numpy as np
from unittest.mock import MagicMock
from src.processing.video_processor import VideoProcessor
from src.detection.detector import Detection


class TestVideoProcessor(unittest.TestCase):
    def setUp (self):
        # Create a mock detector
        self.detector = MagicMock()
        self.detector.detect.return_value = [
            Detection(
                bbox=(100, 100, 200, 200),
                class_id=1,
                class_name="test",
                confidence=0.9
            )
        ]

        # Create the processor with mock detector
        self.processor = VideoProcessor(
            self.detector,
            {
                "detection_interval": 2,
                "tracker_type": "KCF"
            }
        )

        # Create a test frame
        self.frame = np.zeros((300, 300, 3), dtype=np.uint8)

    def test_initialization (self):
        """Test processor initialization"""
        self.assertEqual(self.processor.detection_interval, 2)
        self.assertEqual(self.processor.frame_count, 0)
        self.assertEqual(self.processor.fps, 0)

    def test_process_frame_detection (self):
        """Test processing a frame with detection"""
        # First frame should trigger detection
        _, detections = self.processor.process_frame(self.frame)

        # Detector should have been called
        self.detector.detect.assert_called_once()

        # Frame count should be incremented
        self.assertEqual(self.processor.frame_count, 1)

        # FPS should be calculated
        self.assertGreater(self.processor.fps, 0)

    def test_detection_interval (self):
        """Test that detection runs at the configured interval"""
        # Process first frame (should detect)
        self.processor.process_frame(self.frame)
        self.detector.detect.reset_mock()

        # Process second frame (should not detect)
        self.processor.process_frame(self.frame)
        self.detector.detect.assert_not_called()

        # Process third frame (should detect again)
        self.processor.process_frame(self.frame)
        self.detector.detect.assert_called_once()


if __name__ == '__main__':
    unittest.main()
