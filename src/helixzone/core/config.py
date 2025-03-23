"""Configuration system for HelixZone."""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "app": {
        "name": "HelixZone",
        "version": "0.1.0",
        "temp_dir": "temp",
    },
    "ui": {
        "theme": "system",
        "language": "en",
        "toolbar_size": "medium",
        "icon_theme": "default"
    },
    "editing": {
        "undo_levels": 50,
        "default_format": "png"
    },
    "performance": {
        "tile_size_mb": 100,
        "max_threads": 4,
        "use_gpu": True
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the application configuration.
    
    Args:
        config_path: Optional path to a JSON configuration file
        
    Returns:
        Loaded configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # If a config path is provided, try to load it
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Update the default config with user settings
            _update_config_recursive(config, user_config)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    return config

def _update_config_recursive(base_config: Dict[str, Any], update_config: Dict[str, Any]) -> None:
    """Recursively update a configuration dictionary.
    
    Args:
        base_config: Base configuration to update
        update_config: Update values
    """
    for key, value in update_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            # Recursively update nested dictionaries
            _update_config_recursive(base_config[key], value)
        else:
            # Directly update values
            base_config[key] = value

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """Save the configuration to a file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save to
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")
        return False 