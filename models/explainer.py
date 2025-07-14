import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation
    for visualizing CNN decision-making process.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for gradient computation
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, 
                     input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index (if None, use predicted class)
            
        Returns:
            CAM heatmap as numpy array
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # (H, W)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize CAM
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def generate_heatmap(self, 
                        input_tensor: torch.Tensor,
                        target_class: Optional[int] = None,
                        alpha: float = 0.4,
                        colormap: str = 'jet') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate heatmap overlay on original image.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index
            alpha: Overlay transparency
            colormap: Matplotlib colormap name
            
        Returns:
            Tuple of (cam, heatmap_overlay)
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class)
        
        # Resize CAM to input size
        input_size = input_tensor.shape[2:]  # (H, W)
        cam_resized = cv2.resize(cam, (input_size[1], input_size[0]))
        
        # Convert input tensor to image
        input_image = self._tensor_to_image(input_tensor)
        
        # Apply colormap to CAM
        colormap_func = cm.get_cmap(colormap)
        cam_colored = colormap_func(cam_resized)
        cam_colored = (cam_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Create overlay
        heatmap_overlay = cv2.addWeighted(
            input_image, 1 - alpha, cam_colored, alpha, 0
        )
        
        return cam_resized, heatmap_overlay
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to image array.
        
        Args:
            tensor: Input tensor (1, C, H, W)
            
        Returns:
            Image as numpy array (H, W, C)
        """
        # Remove batch dimension and convert to numpy
        image = tensor.squeeze(0).detach().cpu().numpy()  # (C, H, W)
        
        # Transpose to (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        
        # Denormalize (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Clip and convert to uint8
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        
        return image

class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation for improved localization.
    """
    
    def generate_cam(self, 
                     input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index
            
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate alpha weights (Grad-CAM++)
        alpha_num = gradients.pow(2)
        alpha_denom = 2.0 * gradients.pow(2) + \
                     (activations * gradients.pow(3)).sum(dim=(1, 2), keepdim=True)
        
        alpha = alpha_num / (alpha_denom + 1e-7)
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()

class DefectExplainer:
    """
    High-level interface for defect explanation using Grad-CAM.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 class_names: dict,
                 device: torch.device,
                 explainer_type: str = "gradcam"):
        """
        Initialize defect explainer.
        
        Args:
            model: Trained defect classifier
            class_names: Mapping of class indices to names
            device: Computation device
            explainer_type: Type of explainer ('gradcam' or 'gradcam++')
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.explainer_type = explainer_type
        
        # Get target layer
        target_layer = self._get_target_layer()
        
        # Initialize explainer
        if explainer_type == "gradcam":
            self.explainer = GradCAM(model, target_layer)
        elif explainer_type == "gradcam++":
            self.explainer = GradCAMPlusPlus(model, target_layer)
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")
    
    def _get_target_layer(self) -> torch.nn.Module:
        """Get the appropriate target layer for the model."""
        return self.model.get_target_layer()
    
    def explain_prediction(self, 
                          input_tensor: torch.Tensor,
                          target_class: Optional[int] = None,
                          alpha: float = 0.4,
                          colormap: str = 'jet',
                          return_image: bool = True) -> dict:
        """
        Generate explanation for model prediction.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class for explanation
            alpha: Heatmap overlay transparency
            colormap: Matplotlib colormap
            return_image: Whether to return PIL Image
            
        Returns:
            Dictionary containing explanation results
        """
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        # Generate heatmap
        cam, heatmap_overlay = self.explainer.generate_heatmap(
            input_tensor, target_class, alpha, colormap
        )
        
        # Prepare results
        results = {
            'predicted_class': predicted_class,
            'predicted_class_name': self.class_names.get(predicted_class, f"Class_{predicted_class}"),
            'confidence': confidence,
            'probabilities': {
                self.class_names.get(i, f"Class_{i}"): float(prob) 
                for i, prob in enumerate(probabilities[0])
            },
            'cam': cam,
            'heatmap_array': heatmap_overlay
        }
        
        # Convert to PIL Image if requested
        if return_image:
            results['heatmap_image'] = Image.fromarray(heatmap_overlay)
        
        return results
    
    def batch_explain(self, 
                     input_batch: torch.Tensor,
                     target_classes: Optional[list] = None,
                     alpha: float = 0.4) -> list:
        """
        Generate explanations for a batch of images.
        
        Args:
            input_batch: Input batch tensor (B, C, H, W)
            target_classes: List of target classes
            alpha: Heatmap transparency
            
        Returns:
            List of explanation dictionaries
        """
        results = []
        
        for i in range(input_batch.size(0)):
            single_input = input_batch[i:i+1]  # Keep batch dimension
            target_class = target_classes[i] if target_classes else None
            
            explanation = self.explain_prediction(
                single_input, target_class, alpha
            )
            results.append(explanation)
        
        return results
    
    def save_explanation(self, 
                        explanation: dict, 
                        save_path: str,
                        show_confidence: bool = True):
        """
        Save explanation visualization to file.
        
        Args:
            explanation: Explanation dictionary
            save_path: Output file path
            show_confidence: Whether to show confidence in title
        """
        if 'heatmap_image' in explanation:
            image = explanation['heatmap_image']
            
            # Add title with prediction info
            if show_confidence:
                title = f"Predicted: {explanation['predicted_class_name']} " \
                       f"(Confidence: {explanation['confidence']:.2%})"
                
                # Create figure with title
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                ax.imshow(image)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                image.save(save_path)
        else:
            raise ValueError("No heatmap image found in explanation")

def create_explainer(model: torch.nn.Module, 
                    class_names: dict,
                    device: torch.device,
                    explainer_type: str = "gradcam") -> DefectExplainer:
    """
    Factory function to create defect explainer.
    
    Args:
        model: Trained model
        class_names: Class name mapping
        device: Computation device
        explainer_type: Explainer type
        
    Returns:
        DefectExplainer instance
    """
    return DefectExplainer(model, class_names, device, explainer_type)
