# pruning_toolkit/config.py
import json
import os
import logging

logger = logging.getLogger(__name__)

class PruningConfig:
    """
    Manages configuration for the pruning toolkit, loaded from a JSON file.
    """
    def __init__(self, config_path=None):
        self.config = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        """Loads configuration from a specified JSON file."""
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded successfully from {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {config_path}: {e}")
            raise ValueError(f"Invalid JSON format in {config_path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading config from {config_path}: {e}")
            raise

    def get(self, key, default=None):
        """Retrieves a configuration value by key, with an optional default."""
        # Split key by '.' for nested dictionary access
        keys = key.split('.')
        current_level = self.config
        for k in keys:
            if isinstance(current_level, dict) and k in current_level:
                current_level = current_level[k]
            else:
                return default
        return current_level

    def __getitem__(self, key):
        """Allows dictionary-like access to configuration values."""
        value = self.get(key)
        if value is None and not key.startswith("quantization."): # Don't warn for deep quantization keys
            logger.warning(f"Configuration key '{key}' not found. Returning None.")
        return value

    def __str__(self):
        return json.dumps(self.config, indent=2)