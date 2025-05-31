import argparse
import logging
from pathlib import Path

import cv2

from src.video_pipeline.frame_processor import FrameProcessor
from src.video_pipeline.video_source import VideoSource

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def display_metadata(metadata):
    """Display metadata in a formatted way."""
    logger.info("Video Metadata:")
    for key, value in metadata.items():
        logger.info(f"  {key}: {value}")

def process_video(source_path, output_dir=None, sample_rate=30, max_frames=100):
    """
    Process a video file or camera feed and display frames.

    Args:
        source_path: Path to video file or camera index
        output_dir: Directory to save processed frames
        sample_rate: Process every Nth frame
        max_frames: Maximum number of frames to process
    """
    video_source = VideoSource(source=source_path)
    if not video_source.open():
        logger.error(f"Failed to open video source: {source_path}")
        return

    # Display metadata
    metadata = video_source.get_metadata()
    display_metadata(metadata)

    # Initialize frame processor
    frame_processor = FrameProcessor(
        target_resolution=(640, 480),
        color_format='RGB',
        normalize=False
    )

    # Extract frames with timestamps
    logger.info(f"Extracting frames (sample rate: {sample_rate}, max frames: {max_frames})...")
    frames, timestamps = frame_processor.extract_frames(
        video_source,
        sample_rate=sample_rate,
        max_frames=max_frames,
        return_timestamps=True
    )

    logger.info(f"Extracted {len(frames)} frames")

    # Save frames if output directory is specified
    if output_dir and frames:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Saving frames to {output_dir}")
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            filename = f"frame_{i:04d}_{timestamp:.2f}s.jpg"
            frame_processor.save_frame(frame, str(output_dir), filename)

    # Display frames
    if frames:
        logger.info("Displaying frames (press 'q' to exit)...")
        for i, frame in enumerate(frames):
            # Convert back to BGR for OpenCV display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Add frame number and timestamp
            timestamp = timestamps[i] if timestamps else i
            cv2.putText(
                display_frame,
                f"Frame {i}, Time: {timestamp:.2f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Video Frame", display_frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    video_source.release()
    cv2.destroyAllWindows()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Video Analytics Dashboard Demo")
    parser.add_argument("--source", "-s", type=str, default="0",
        help="Video file path or camera index (default: 0)")
    parser.add_argument("--output", "-o", type=str, default=None,
        help="Output directory for saved frames")
    parser.add_argument("--sample_rate", "-r", type=int, default=30,
        help="Proces every Nth frame (default: 30)")
    parser.add_argument("--max-frames", "-m", type=int, default=100,
        help="Maximum number of frames to process (default: 100)")

    args = parser.parse_args()

    # Handle numeric camera index
    source = args.source
    if source.isdigit():
        source = int(source)

    process_video(source, args.output, args.sample_rate, args.max_frames)

if __name__ == "__main__":
    main()
