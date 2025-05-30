import unittest
import numpy as np
import cv2
from src.video_pipeline.frame_processor import FrameProcessor
from src.video_pipeline.video_source import VideoSource

class TestFrameProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FrameProcessor(target_resolution=(320, 240), color_format='RGB', normalize=True)
        self.mock_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        self.mock_gray_frame = cv2.cvtColor(self.mock_frame, cv2.COLOR_BGR2GRAY)

    def test_process_frame_resize(self):
        processed_frame = self.processor.process_frame(self.mock_frame, apply_resize=True)
        self.assertEqual(processed_frame.shape[:2], (240, 320))

    def test_process_frame_color_conversion(self):
        processed_frame = self.processor.process_frame(self.mock_frame, apply_color_conversion=True)
        self.assertEqual(processed_frame.shape[2], 3)  # RGB format

    def test_process_frame_normalization(self):
        processed_frame = self.processor.process_frame(self.mock_frame, apply_normalization=True)
        # Check that normalization happened (values were transformed to float)
        self.assertEqual(processed_frame.dtype, np.float32)
        # Standardized values typically fall within a reasonable range
        self.assertTrue(np.all(processed_frame > -5.0) and np.all(processed_frame < 5.0))

    def test_buffer_frames(self):
        self.processor.buffer_frames(self.mock_frame)
        self.assertEqual(len(self.processor.frame_buffer), 1)

    def test_get_frame_difference(self):
        diff = self.processor.get_frame_difference(self.mock_frame, self.mock_frame, threshold=10)
        self.assertTrue(np.all(diff == 0))  # No difference between identical frames

    def test_extract_frames(self):
        video_source = VideoSource(source=0)  # Mock camera input
        video_source.open = lambda: True  # Mock open method
        video_source.read = lambda: (True, self.mock_frame)  # Mock read method
        frames = self.processor.extract_frames(video_source, sample_rate=1, max_frames=5)
        self.assertEqual(len(frames), 5)

if __name__ == '__main__':
    unittest.main()