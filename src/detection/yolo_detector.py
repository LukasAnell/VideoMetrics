import logging
import os
from typing import Dict, Any, List

import cv2
import numpy as np

from .config import DetectionConfig
from .detector import Detector, Detection

logger = logging.getLogger(__name__)

class YOLODetector(Detector):
    """YOLO-based object detector using OpenCV DNN"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize YOLO detector with configuration.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.config = DetectionConfig(config)
        self.net = None
        self.output_layers = []
        self.input_width = self.config.get('input_width', 416)
        self.input_height = self.config.get('input_height', 416)
        self.scale = self.config.get('input_scale', 1.0 / 255.0)
        self.swap_rb = self.config.get('swap_rb', True)
        self.nms_threshold = self.config.get('nms_threshold', 0.45)

    def load_model(self, model_path: str) -> bool:
        """
        Initialize YOLO detector with configuration.

        Args:
            config: Configuration dictionary
        """
        try:
            # Look for configuration files
            if os.path.isdir(model_path):
                weights_files = [f for f in os.listdir(model_path) if f.endswith('.weights')]
                cfg_files = [f for f in os.listdir(model_path) if f.endswith('.cfg')]
                names_files = [f for f in os.listdir(model_path) if f.endswith('.names')]

                if not weights_files or not cfg_files:
                    logger.error(f"Missing model files in {model_path}")
                    return False

                weights_path = os.path.join(model_path, weights_files[0])
                cfg_path = os.path.join(model_path, cfg_files[0])

                if names_files:
                    names_path = os.path.join(model_path, names_files[0])
                    self._load_classes(names_path)
            else:
                # Assume model_path is the weights file
                weights_path = model_path
                cfg_path = model_path.replace('.weights', '.cfg')
                names_path = model_path.replace('.weights', '.names')

                if os.path.exists(names_path):
                    self._load_classes(names_path)

            # Load network
            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

            # Set preferred backend and target
            if self.config.get('device') == 'cuda':
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            self.initialized = True
            logger.info(f"Loaded YOLO model: {weights_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            return False

    def _load_classes(self, names_path: str):
        """Load class names from file"""
        try:
            with open(names_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.classes)} classes from {names_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading class names: {str(e)}")
            self.classes = []
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of Detection objects
        """
        if not self.initialized or self.net is None:
            logger.error("Model not initialized")
            return []

        if frame is None:
            return []

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            frame,
            self.scale,
            (self.input_width, self.input_height),
            swapRB=self.swap_rb
        )

        # Forward pass
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # Process outputs
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    # YOLO returns coordinates relative to the center
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x1 = max(0, int(center_x - w / 2))
                    y1 = max(0, int(center_y - h / 2))
                    x2 = min(width, int(center_x + w / 2))
                    y2 = min(height, int(center_y + h / 2))

                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )

        # Create Detection objects
        detections = []
        for i in indices:
            # OpenCV 4.5.4+ returns a single-dimensional array
            if isinstance(i, (tuple, list)):
                i = i[0]

            box = boxes[i]
            class_id = class_ids[i]

            class_name = "unknown"
            if 0 <= class_id < len(self.classes):
                class_name = self.classes[class_id]

            detections.append(Detection(
                bbox=box,
                class_id=class_id,
                class_name=class_name,
                confidence=confidences[i]
            ))

        return detections
