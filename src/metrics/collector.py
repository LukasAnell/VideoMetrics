import time
from typing import List, Tuple, Dict

import cv2
import numpy as np

from src.detection.detector import Detection


class MetricsCollector:
    """Collects and analyzes metrics from tracked objects"""

    def __init__(self):
        # Object counts by class
        self.class_counts = {}
        # Track IDs we've seen
        self.tracked_ids = set()
        # Store entry time for each track
        self.track_entry_times = {}
        # Store positions for trajectory analysis
        self.track_positions = {}
        # Track time in frame
        self.time_in_frame = {}
        # Frame count
        self.frame_count = 0
        # Start time
        self.start_time = time.time()

        self.zones = {} # Zone name -> (x1, y1, x2, y2)
        self.zone_counts = {} # Zone name -> count of objects entered

    def add_zone(self, name: str, bbox: Tuple[int, int, int, int]):
        """Add a zone for analytics"""
        self.zones[name] = bbox
        self.zone_counts[name] = 0

    def update(self, tracked_objects: List[Detection], frame_shape: Tuple[int, int, int]):
        """Update metrics with new tracked objects data"""
        self.frame_count += 1
        current_time = time.time()
        current_tracks = set()

        height, width = frame_shape[:2]

        # Process each tracked object
        for obj in tracked_objects:
            track_id = obj.track_id
            if track_id is None: continue

            # Count by class
            if obj.class_name not in self.class_counts:
                self.class_counts[obj.class_name] = 0

            # New object detected
            if track_id not in self.tracked_ids:
                self.tracked_ids.add(track_id)
                self.class_counts[obj.class_name] += 1
                self.track_entry_times[track_id] = current_time
                self.track_positions[track_id] = []
                self.time_in_frame[track_id] = 0

            # Record current position (center of bounding box)
            x1, y1, x2, y2 = obj.bbox
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            self.track_positions[track_id].append((center_x, center_y))

            # Update time in frame
            if track_id in self.track_entry_times:
                self.time_in_frame[track_id] = current_time - self.track_entry_times[track_id]

            current_tracks.add(track_id)

            # Check zone entry
            center_x, center_y = self.track_positions[track_id][-1]
            for zone_name, (x1, y1, x2, y2) in self.zones.items():
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    self.zone_counts[zone_name] += 1

        # Clean up tracks that are no longer present
        for track_id in list(self.track_entry_times.keys()):
            if track_id not in current_tracks:
                # Could add "exit event" processing here
                pass

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw zones on the frame"""
        for zone_name, (x1, y1, x2, y2) in self.zones.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{zone_name}: {self.zone_counts[zone_name]}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
        return frame

    def get_summary(self) -> Dict:
        """Get summary of collected metrics"""
        return {
            "total_tracked": len(self.tracked_ids),
            "class_counts": self.class_counts,
            "current_in_frame": len(self.time_in_frame),
            "avg_time_in_frame": sum(self.time_in_frame.values()) / max(len(self.time_in_frame), 1),
            "runtime": time.time() - self.start_time,
            "processed_frames": self.frame_count
        }

    def draw_metrics(self, frame: np.ndarray) -> np.ndarray:
        """Draw metrics overlay on frame"""
        frame = self.draw_zones(frame) # Draw zones
        # Draw summary text
        y_pos = 30
        frame = cv2.putText(
            frame,
            f"Total Tracked: {len(self.tracked_ids)}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Class counts
        for i, (class_name, count) in enumerate(self.class_counts.items()):
            y_pos += 30
            frame = cv2.putText(
                frame,
                f"{class_name}: {count}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Draw trajectories for current objects
        for track_id, positions in self.track_positions.items():
            if len(positions) > 1:
                # Draw trajectory line
                for i in range(1, len(positions)):
                    cv2.line(
                        frame,
                        positions[i - 1],
                        positions[i],
                        (0, 0, 255),
                        2
                    )
        return frame
