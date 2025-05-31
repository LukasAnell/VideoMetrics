import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

class DetectionConfig:
    """Configuratoin handler for object detection"""

    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration with default values.

        Args:
            config_dict: Optional configuration dictionary to override defaults
        """
        # Default configuration
        self.config = {
            # Model settings
            'model_path': '',
            'model_type': 'yolo',  # 'yolo', 'ssd', etc.
            'device': 'cpu',  # 'cpu', 'cuda', etc.

            # Detection parameters
            'confidence_threshold': 0.5,
            'nms_threshold': 0.45,
            'class_filter': [],  # Empty list means all classes

            # Input processing
            'input_width': 416,
            'input_height': 416,
            'input_scale': 1.0/255.0,
            'swap_rb': True,

            # Additional options
            'enable_tracking': False,
            'max_detections': 100,
        }

        # Override defaults with provided config
        if config_dict:
            self.config.update(config_dict)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key"""
        return self.config.get(key)

    def __setitem__(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value

    @classmethod
    def from_file(cls, config_path: str) -> 'DetectionConfig':
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration JSON file

        Returns:
            DetectionConfig instance
        """
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls(config_dict)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return cls()

    def save(self, config_path: str) -> bool:
        """
        Save configuration to JSON file.

        Args:
            config_path: Path to save configuration

        Returns:
            True if saved successfully
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default fallback"""
        return self.config.get(key, default)
