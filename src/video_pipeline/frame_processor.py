import logging
import os
from collections import deque
from typing import Tuple, Dict, Union, List, Callable

import cv2
import numpy as np

from src.video_pipeline.video_source import VideoSource

logger = logging.getLogger(__name__)

class FrameProcessor:
    """
    FrameProcessor processes video frames for various tasks such as resizing, cropping, and normalization.
    Provides methods to apply transformations and prepare frames for model inference.
    """

    def __init__(self, target_resolution: Tuple[int, int] = (640, 480),
            color_format: str = 'BGR',
            normalize: bool = False,
            buffer_size: int = 30,
            preprocessing_options: Dict = None):
        """
        Initialize a FrameProcessor.

        Args:
            target_resolution: Desired resolution for the frames (width, height)
            color_format: Color format for the frames (e.g., 'RGB', 'BGR')
            normalize: Whether to normalize pixel values
            buffer_size: Number of frames to buffer
            preprocessing_options: Additional preprocessing options (e.g., cropping, resizing)
        """
        self.target_resolution = target_resolution
        self.color_format = color_format
        self.normalize = normalize
        self.buffer_size = buffer_size
        self.preprocessing_options = preprocessing_options or {}
        self.frame_buffer = deque(maxlen=buffer_size)

        valid_formats = ['BGR', 'RGB', 'GRAY']
        if self.color_format not in valid_formats:
            logger.warning(f"Invalid color format '{self.color_format}'. Defaulting to 'BGR'.")
            self.color_format = 'BGR'

    def process_frame(self, frame: np.ndarray,
            apply_resize: bool = True,
            apply_color_conversion: bool = True,
            apply_normalization: bool = None,
            additional_transforms: List = None) -> np.ndarray:
        """
        Process a single frame with the configured transformations.

        Args:
            frame: Input frame to process
            apply_resize: Whether to resize the frame
            apply_color_conversion: Whether to convert color space
            apply_normalization: Whether to normalize the frame (overrides class setting if provided)
            additional_transforms: Custom transformations to apply

        Returns:
            Processed frame
        """
        if frame is None:
            logger.warning("Received None frame for processing.")
            return None

        processed_frame = frame.copy()

        # Apply resizing if requested
        if apply_resize:
            processed_frame = resize_frame(processed_frame, self.target_resolution)

        # Apply color conversion if requested
        if apply_color_conversion:
            # Detect source format based on channels
            source_format = 'BGR' # OpenCV default
            if len(processed_frame.shape) == 2:
                source_format = 'GRAY'

            if source_format != self.color_format:
                processed_frame = convert_color(processed_frame, source_format, self.color_format)

        # Apply normalization if requested
        if additional_transforms:
            processed_frame = self.apply_transformations(
                processed_frame,
                additional_transforms,
                {}
            )
        return processed_frame

    def extract_frames(self, video_source: VideoSource,
            sample_rate: int = 1,
            max_frames: int = None,
            start_time: float = None,
            end_time: float = None,
            return_timestamps: bool = False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[float]]]:
        """
        Extract frames from a video source with various sampling options.

        Args:
            video_source: Source to extract frames from
            sample_rate: Extract every Nth frame
            max_frames: Maximum number of frames to extract
            start_time: Time in seconds to start extraction
            end_time: Time in seconds to end extraction
            return_timestamps: Whether to return frame timestamps

        Returns:
            If return_timestamps is False: List of extracted frames
            If return_timestamps is True: Tuple of (frames, timestamps)
        """
        if not video_source.is_opened:
            if not video_source.open():
                logger.error("Could not open video source for frame extraction")
                return ([], []) if return_timestamps else []

        metadata = video_source.get_metadata()
        fps = metadata.get("fps", 30) # Default to 30 FPS if not available

        # Calculate frame posistions for time ranges
        frame_positions = []
        if start_time is not None:
            start_frame = int(start_time * fps)
            if isinstance(video_source.source, str): # Only seek in file-based videos
                video_source.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            start_frame = 0

        end_frame = None
        if end_time is not None and fps > 0:
            end_frame = int(end_time * fps)

        # Extract frames
        frames = []
        timestamps = []
        frame_count = 0
        current_frame = start_frame

        while True:
            # Check if we've reached max frames
            if max_frames is not None and frame_count >= max_frames:
                break

            # Check if we've reached end time
            if end_frame is not None and current_frame >= end_frame:
                break

            # Read the frame
            if current_frame % sample_rate == 0:
                ret, frame = video_source.read()
                if not ret: break

                processed_frame = self.process_frame(frame)
                frames.append(processed_frame)

                if return_timestamps:
                    timestamp = current_frame / fps
                    timestamps.append(timestamp)

                frame_count += 1
            else:
                # Skip frame but advance position
                ret, _ = video_source.read()
                if not ret: break

        return (frames, timestamps) if return_timestamps else frames

    def buffer_frames(self, frame: np.ndarray, metadata: Dict = None):
        """
        Add a frame to the buffer.

        Args:
            frame: Frame to add to buffer
            metadata: Optional metadata for the frame
        """
        if frame is None: return

        frame_data = {'frame': frame}
        if metadata:
            frame_data.update({'metadata': metadata})

        self.frame_buffer.append(frame_data)

    def get_frame_difference(self, frame1: np.ndarray,
            frame2: np.ndarray,
            threshold: float = 25.0,
            method: str = 'absdiff') -> np.ndarray:
        """
        Calculate difference between two frames.

        Args:
            frame1: First frame
            frame2: Second frame
            threshold: Difference detection threshold
            method: Method for calculating difference ('absdiff' or 'subtract')

        Returns:
            Difference mask
        """
        if frame1 is None or frame2 is None:
            logger.warning("Cannot calculate frame difference with None frames.")
            return None

        # Ensure frames have the same size and color format
        if frame1.shape != frame2.shape:
            logger.warning("Frames have different shapes, resizing frame2")
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Convert to graysale if they're color images
        if len(frame1.shape) == 3:
            gray1 = convert_color(frame1, 'BGR', 'GRAY')
            gray2 = convert_color(frame2, 'BGR', 'GRAY')
        else:
            gray1 = frame1
            gray2 = frame2

        # Calculate difference
        if method == 'absdiff':
            diff = cv2.absdiff(gray1, gray2)
        else:
            diff = cv2.subtract(gray1, gray2)

        # Apply threshold
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        return mask

    def apply_transformations(self, frame: np.ndarray,
            transformations: List,
            params: Dict = None) -> np.ndarray:
        """
        Apply a list of transformations to a frame.

        Args:
            frame: Frame to transform
            transformations: List of transformation functions or names
            params: Parameters for each transformation

        Returns:
            Transformed frame
        """
        if frame is None:
            return None

        params = params or {}
        result = frame.copy()

        for transform in transformations:
            if callable(transform):
                # If the transform is a function, call it directly
                result = transform(result, **params.get(transform.__name__, {}))
            elif isinstance(transform, str):
                # Handle built-in transformations
                if transform == 'flip_horizontal':
                    result = cv2.flip(result, 1)
                elif transform == 'flip_vertical':
                    result = cv2.flip(result, 0)
                elif transform == 'rotate_90':
                    result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
                elif transform == 'rotate_180':
                    result = cv2.rotate(result, cv2.ROTATE_180)
                elif transform == 'rotate_270':
                    result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif transform == 'blur':
                    ksize = params.get('blur', {}).get('ksize', (5, 5))
                    result = cv2.GaussianBlur(result, ksize, 0)
                else:
                    logger.warning(f"Unknown transformation '{transform}'")
            else:
                logger.warning(f"Invalid transformation type: {type(transform)}")

        return result

    def batch_process(self, frames: List[np.ndarray],
            batch_size: int = 8,
            processing_fn: Callable = None) -> List[np.ndarray]:
        """
        Process a batch of frames for efficiency.

        Args:
            frames: List of frames to process
            batch_size: Size of each processing batch
            processing_fn: Function to apply to each batch (defaults to self.process_frame)

        Returns:
            List of processed frames
        """
        if not frames:
            return []

        if processing_fn is None:
            procecssing_fn = self.process_frame

        results = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]

            # Process batch
            processed_batch = [processing_fn(frame) for frame in batch]
            results.extend(processed_batch)

        return results

    def save_frame(self, frame: np.ndarray,
            output_path: str,
            filename: str,
            quality: int = 95) -> bool:
        """
        Save a frame to disk.

        Args:
            frame: Frame to save
            output_path: Directory to save the frame
            filename: Name for the saved file
            quality: JPEG quality (0-100)

        Returns:
            Boolean indicating success
        """
        if frame is None:
            logger.warning("Cannot save None frame.")
            return False

        # Create directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Ensure filename has extension
        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            filename += '.jpg'

        full_path = os.path.join(output_path, filename)

        try:
            success = cv2.imwrite(full_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if success:
                logger.debug(f"Saved frame to {full_path}")
            else:
                logger.error(f"Failed to save frame to {full_path}")
            return success
        except Exception as e:
            logger.error(f"Error saving frame: {str(e)}")
            return False

    def frame_to_tensor(self, frame: np.ndarray,
            data_format: str = 'channels_last',
            device: str = 'cpu') -> np.ndarray:
        """
        Convert a frame to tensor format for ML models.

        Args:
            frame: Frame to convert
            data_format: Format for the output tensor ('channels_first' or 'channels_last')
            device: Target device for tensor ('cpu', 'cuda') - for future compatibility

        Returns:
            Tensor-ready numpy array
        """
        if frame is None:
            return None

        # Ensure frame is in float format for ML models
        if frame.dtype != np.ndarray:
            tensor = frame.astype(np.float32)
        else:
            tensor = frame.copy()

        # Normalize if not already done
        if tensor.max() > 1.0:
            tensor /= 255.0

        # Handle channel format
        if data_format == 'channels_first' and len(tensor.shape) == 3:
            # Conert from HWC to CHW
            tensor = np.transpose(tensor, (2, 0, 1))

        # Add batch dimension if needed
        if len(tensor.shape) == 3:
            tensor = np.expand_dims(tensor, axis=0)

        return tensor


# Helper functions

def resize_frame(frame: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize a frame to target dimensions.

    Args:
        frame: Frame to resize
        target_size: Width and height
        interpolation: OpenCV interpolation method

    Returns:
        Resized frame
    """
    if frame is None:
        return None

    return cv2.resize(frame, target_size, interpolation=interpolation)

def normalize_frame(frame: np.ndarray,
        mean: List[float] = None,
        std: List[float] = None,
        to_float: bool = True) -> np.ndarray:
    """
    Normalize pixel values for ML processing.

    Args:
        frame: Frame to normalize
        mean: Channel means (defaults to [0.485, 0.456, 0.406] for RGB)
        std: Channel standard deviations (defaults to [0.229, 0.224, 0.225] for RGB)
        to_float: Convert to float before normalization

    Returns:
        Normalized frame
    """
    if frame is None:
        return None

    # Default normalization values (ImageNet)
    if mean is None:
        mean = [0.485, 0.456, 0.406] if len(frame.shape) == 3 else [0.5]
    if std is None:
        std = [0.229, 0.224, 0.225] if len(frame.shape) == 3 else [0.5]

    # Convert to float
    if to_float and frame.dtype != np.float32:
        normalized = frame.astype(np.float32) / 255.0
    else:
        normalized = frame.astype(np.float32)

    # Apply normalization
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Multi-channel image
        for i in range(min(len(mean), frame.shape[2])):
            normalized[:, :, i] = (normalized[:, :, i] - mean[i]) / std[i]
    else:
        # Single-channel image
        normalized = (normalized - mean[0]) / std[0]

    return normalized

def convert_color(frame: np.ndarray,
        source_format: str,
        target_format: str) -> np.ndarray:
    """
    Convert between color spaces.

    Args:
        frame: Frame to convert
        source_format: Source color format ('BGR', 'RGB', 'GRAY')
        target_format: Target color format ('BGR', 'RGB', 'GRAY')

    Returns:
        Converted frame
    """
    if frame is None:
        return None

    if source_format == target_format:
        return frame

    # Define conversion mappings
    conversion_map = {
        ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
        ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
        ('BGR', 'GRAY'): cv2.COLOR_BGR2GRAY,
        ('RGB', 'GRAY'): cv2.COLOR_RGB2GRAY,
        ('GRAY', 'BGR'): cv2.COLOR_GRAY2BGR,
        ('GRAY', 'RGB'): cv2.COLOR_GRAY2RGB
    }

    conversion_key = (source_format, target_format)
    if conversion_key in conversion_map:
        return cv2.cvtColor(frame, conversion_map[conversion_key])
    else:
        logger.warning(f"Unsupported color conversion: {source_format} to {target_format}")
        return frame

def annotate_frame(frame: np.ndarray,
        text: str,
        position: Tuple[int, int] = (10, 30),
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2) -> np.ndarray:
    """
    Add text annotation to a frame.

    Args:
        frame: Frame to annotate
        text: Text to add
        position: (x, y) position
        font: OpenCV font type
        color: (B, G, R) color
        thickness: Text thickness

    Returns:
        Annotated frame
    """
    if frame is None:
        return None

    annotated = frame.copy()
    cv2.putText(
        annotated,
        text,
        position,
        font,
        0.7,
        color,
        thickness,
        cv2.LINE_AA
    )

    return annotated
