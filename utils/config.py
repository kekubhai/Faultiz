import yaml
import os
from typing import Dict, Any
import logging

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for config.yaml in project root.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Find project root and config file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, "config.yaml")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Validate and set defaults
        config = _set_config_defaults(config)
        
        logging.info(f"Configuration loaded from {config_path}")
        return config
        
    except FileNotFoundError:
        logging.warning(f"Config file not found: {config_path}. Using defaults.")
        return _get_default_config()
    
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}. Using defaults.")
        return _get_default_config()

def _set_config_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Set default values for missing configuration keys."""
    
    defaults = _get_default_config()
    
    # Merge with defaults
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(config[key], dict):
            # Recursively merge nested dictionaries
            config[key] = {**default_value, **config[key]}
    
    return config

def _get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    
    return {
        'model': {
            'name': 'efficientnet-b0',
            'num_classes': 7,
            'pretrained': True,
            'checkpoint_path': None
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 50,
            'image_size': 224,
            'data_augmentation': True
        },
        'defect_classes': {
            0: "Normal",
            1: "Scratch", 
            2: "Dent",
            3: "Crack",
            4: "Corrosion",
            5: "Missing_Part",
            6: "Color_Defect"
        },
        'google_cloud': {
            'credentials_path': None,
            'project_id': None
        },
        'llm': {
            'model_name': 'microsoft/DialoGPT-medium',
            'max_length': 512,
            'temperature': 0.7,
            'device': 'auto'
        },
        'streamlit': {
            'page_title': 'Faultiz - Visual Defect Classifier',
            'page_icon': '🔍',
            'layout': 'wide',
            'sidebar_state': 'expanded'
        },
        'image_processing': {
            'max_upload_size': 10,
            'supported_formats': ['jpg', 'jpeg', 'png', 'bmp'],
            'resize_method': 'bilinear'
        },
        'gradcam': {
            'target_layer': 'features.7',
            'alpha': 0.4,
            'colormap': 'jet'
        },
        'development': {
            'use_mock_ocr': False,
            'use_mock_llm': False,
            'debug_mode': False
        }
    }

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        
        logging.info(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving config: {e}")
        return False

def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model-specific configuration."""
    return config.get('model', {})

def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training-specific configuration."""
    return config.get('training', {})

def get_inference_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract inference-specific configuration."""
    return {
        'image_processing': config.get('image_processing', {}),
        'gradcam': config.get('gradcam', {}),
        'google_cloud': config.get('google_cloud', {}),
        'llm': config.get('llm', {})
    }

def update_config_value(config: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """
    Update a nested configuration value.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the key (e.g., 'model.learning_rate')
        value: New value
        
    Returns:
        Updated configuration dictionary
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value
    
    return config

def validate_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Required sections
    required_sections = ['model', 'defect_classes']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Model validation
    if 'model' in config:
        model_config = config['model']
        
        if 'num_classes' in model_config:
            num_classes = model_config['num_classes']
            if not isinstance(num_classes, int) or num_classes <= 0:
                errors.append("model.num_classes must be a positive integer")
        
        if 'name' in model_config:
            supported_models = ['efficientnet-b0', 'efficientnet-b1', 'resnet50', 'resnet18']
            if model_config['name'] not in supported_models:
                errors.append(f"Unsupported model: {model_config['name']}")
    
    # Defect classes validation
    if 'defect_classes' in config:
        defect_classes = config['defect_classes']
        
        if not isinstance(defect_classes, dict):
            errors.append("defect_classes must be a dictionary")
        else:
            # Check if indices are consecutive integers starting from 0
            indices = list(defect_classes.keys())
            if isinstance(indices[0], str):
                # Convert string keys to integers
                try:
                    indices = [int(k) for k in indices]
                except ValueError:
                    errors.append("defect_classes keys must be integers")
            
            if indices:
                expected_indices = list(range(len(indices)))
                if sorted(indices) != expected_indices:
                    errors.append("defect_classes indices must be consecutive integers starting from 0")
    
    # Training validation
    if 'training' in config:
        training_config = config['training']
        
        if 'image_size' in training_config:
            size = training_config['image_size']
            if not isinstance(size, int) or size <= 0:
                errors.append("training.image_size must be a positive integer")
        
        if 'batch_size' in training_config:
            batch_size = training_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append("training.batch_size must be a positive integer")
    
    return len(errors) == 0, errors

def setup_logging(config: Dict[str, Any] = None):
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    # Default logging configuration
    log_level = logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Override with config if provided
    if config and 'logging' in config:
        logging_config = config['logging']
        
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        log_level = level_map.get(logging_config.get('level', 'INFO'), logging.INFO)
        log_format = logging_config.get('format', log_format)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            # Add file handler if needed
            # logging.FileHandler('faultiz.log')
        ]
    )

def get_device_config() -> str:
    """Get optimal device configuration for the current system."""
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    else:
        return "cpu"
