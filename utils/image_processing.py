import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Union, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

def preprocess_image(image: Image.Image, 
                    target_size: int = 224,
                    normalize: bool = True) -> torch.Tensor:
    """
    Preprocess image for CNN inference.
    
    Args:
        image: PIL Image
        target_size: Target image size
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Preprocessed tensor
    """
    # Define preprocessing transforms
    transform_list = [
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ]
    
    if normalize:
        # ImageNet normalization
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    transform = transforms.Compose(transform_list)
    
    # Apply transforms
    tensor = transform(image)
    
    return tensor

def resize_image(image: Image.Image, 
                max_size: int,
                maintain_aspect: bool = True) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: PIL Image
        max_size: Maximum size for longest dimension
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    if maintain_aspect:
        # Calculate scaling factor
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
    else:
        new_width = new_height = max_size
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def enhance_image(image: Image.Image,
                 contrast: float = 1.2,
                 brightness: float = 1.1,
                 sharpness: float = 1.1) -> Image.Image:
    """
    Enhance image quality for better defect detection.
    
    Args:
        image: PIL Image
        contrast: Contrast enhancement factor
        brightness: Brightness enhancement factor
        sharpness: Sharpness enhancement factor
        
    Returns:
        Enhanced PIL Image
    """
    # Apply enhancements
    enhanced = ImageEnhance.Contrast(image).enhance(contrast)
    enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)
    
    return enhanced

def denoise_image(image: Image.Image, method: str = "gaussian") -> Image.Image:
    """
    Apply denoising to image.
    
    Args:
        image: PIL Image
        method: Denoising method ("gaussian", "median", "bilateral")
        
    Returns:
        Denoised PIL Image
    """
    # Convert to numpy for OpenCV operations
    img_array = np.array(image)
    
    if method == "gaussian":
        denoised = cv2.GaussianBlur(img_array, (5, 5), 0)
    elif method == "median":
        denoised = cv2.medianBlur(img_array, 5)
    elif method == "bilateral":
        denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
    else:
        raise ValueError(f"Unsupported denoising method: {method}")
    
    return Image.fromarray(denoised)

def apply_clahe(image: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: PIL Image
        clip_limit: Clipping limit for CLAHE
        
    Returns:
        CLAHE-enhanced PIL Image
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to LAB color space for better results
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(enhanced)

def detect_edges(image: Image.Image, 
                low_threshold: int = 50,
                high_threshold: int = 150) -> Image.Image:
    """
    Detect edges using Canny edge detection.
    
    Args:
        image: PIL Image
        low_threshold: Low threshold for edge detection
        high_threshold: High threshold for edge detection
        
    Returns:
        Edge-detected image
    """
    # Convert to grayscale numpy array
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Convert back to PIL Image
    return Image.fromarray(edges)

def get_training_transforms(image_size: int = 224,
                           augment: bool = True) -> A.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        image_size: Target image size
        augment: Whether to apply augmentations
        
    Returns:
        Albumentations compose object
    """
    if augment:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

def get_validation_transforms(image_size: int = 224) -> A.Compose:
    """
    Get validation transforms (no augmentation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def create_defect_augmentations() -> A.Compose:
    """
    Create defect-specific augmentations to simulate various defect types.
    
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.OneOf([
            # Scratch simulation
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.1),
                contrast_limit=(0.8, 1.2),
                p=1.0
            ),
            # Corrosion simulation
            A.HueSaturationValue(
                hue_shift_limit=(-10, 5),
                sat_shift_limit=(-20, 10),
                val_shift_limit=(-20, 5),
                p=1.0
            ),
            # General wear simulation
            A.GaussNoise(var_limit=(20.0, 80.0), p=1.0),
        ], p=0.7),
        
        # Geometric augmentations
        A.OneOf([
            A.Rotate(limit=5, p=1.0),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=5,
                p=1.0
            ),
        ], p=0.3),
        
        # Lighting variations
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.RandomGamma(p=1.0),
            A.CLAHE(p=1.0),
        ], p=0.5),
    ])

def normalize_tensor(tensor: torch.Tensor,
                    mean: Optional[List[float]] = None,
                    std: Optional[List[float]] = None) -> torch.Tensor:
    """
    Normalize tensor with given mean and std.
    
    Args:
        tensor: Input tensor
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized tensor
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
    if std is None:
        std = [0.229, 0.224, 0.225]   # ImageNet std
    
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize(tensor)

def denormalize_tensor(tensor: torch.Tensor,
                      mean: Optional[List[float]] = None,
                      std: Optional[List[float]] = None) -> torch.Tensor:
    """
    Denormalize tensor (reverse normalization).
    
    Args:
        tensor: Normalized tensor
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized tensor
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    # Reverse normalization: x = x * std + mean
    denormalized = tensor.clone()
    for i, (m, s) in enumerate(zip(mean, std)):
        denormalized[i] = denormalized[i] * s + m
    
    return denormalized

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Input tensor (C, H, W) or (B, C, H, W)
        
    Returns:
        PIL Image
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize if needed (values should be in [0, 1])
    if tensor.min() < 0:
        tensor = denormalize_tensor(tensor)
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy array
    np_array = tensor.cpu().numpy().transpose(1, 2, 0)
    np_array = (np_array * 255).astype(np.uint8)
    
    return Image.fromarray(np_array)

def pil_to_tensor(image: Image.Image, normalize: bool = False) -> torch.Tensor:
    """
    Convert PIL Image to tensor.
    
    Args:
        image: PIL Image
        normalize: Whether to apply normalization
        
    Returns:
        Tensor (C, H, W)
    """
    transform_list = [transforms.ToTensor()]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    transform = transforms.Compose(transform_list)
    return transform(image)

def calculate_image_stats(image: Image.Image) -> dict:
    """
    Calculate basic statistics of an image.
    
    Args:
        image: PIL Image
        
    Returns:
        Dictionary with image statistics
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    stats = {
        'size': image.size,
        'mode': image.mode,
        'mean_rgb': np.mean(img_array, axis=(0, 1)).tolist(),
        'std_rgb': np.std(img_array, axis=(0, 1)).tolist(),
        'min_value': np.min(img_array),
        'max_value': np.max(img_array),
        'aspect_ratio': image.size[0] / image.size[1]
    }
    
    return stats

def is_image_blurry(image: Image.Image, threshold: float = 100.0) -> bool:
    """
    Detect if image is blurry using Laplacian variance.
    
    Args:
        image: PIL Image
        threshold: Blur threshold
        
    Returns:
        True if image is blurry
    """
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Calculate Laplacian variance
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return variance < threshold
