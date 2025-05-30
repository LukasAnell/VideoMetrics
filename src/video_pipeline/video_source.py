import cv2
import os
import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class VideoSource:
    """
    VideoSource provides a unified interface for different video input types.
    Supports file-based videos and camera feeds.
    """

    def __init__(self, source: Union[str, int], buffer_size: int = 64):
        """
        Initialize a video source.

        Args:
            source: Path to video file (str) or camera index (int)
            buffer_size: Number of frames to buffer (if needed)
            cap: OpenCV VideoCapture object
            is_file: True if source is a file, False if it's a camera index
            is_opened: True if the video source is successfully opened
            metadata: Dictionary to store video metadata
        """
        self.source = source
        self.buffer_size = buffer_size
        self.cap = None
        self.is_file = isinstance(source, str)
        self.is_opened = False
        self.metadata = {}

    def open(self) -> bool:
        """
        Open the video source and extract metadata.

        Returns:
            bool: True if successfully opened, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            self.is_opened = self.cap.isOpened()

            if not self.is_opened:
                logger.error(f"Failed to open video source: {self.source}")
                return False

            # Extract metadata
            self._extract_metadata()
            logger.info(f"Opened video source: {self.source} ({self.metadata}")
            return True

        except Exception as e:
            logger.error(f"Error opening video source: {str(e)}")
            return False

    def _extract_metadata(self) -> None:
        """Extract metadata from the video source."""
        if not self.is_opened:
            return

        # Get basic properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.metadata = {
            "width": width,
            "height": height,
            "fps": fps,
            "resolution": (width, height),
        }

        # For file-based videos, get additional properties
        if self.is_file:
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            self.metadata.update({
                "frame_count": frame_count,
                "duration_sec": duration,
                "filename": os.path.basename(self.source) if isinstance(self.source, str) else None,
            })

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video source.

        Returns:
            Tuple containing:
                bool: True if frame was successfully read
                np.ndarray or None: The frame if successful, None otherwise
        """
        if not self.is_opened:
            if not self.open():
                return False, None
        return self.cap.read()

    def get_metadata(self) -> Dict:
        """
        Get metadata about the video source.

        Returns:
            Dict: Metadata dictionary
        """
        return self.metadata

    def release(self) -> None:
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False

    def __enter__(self):
        """Context manager enter."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
