import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet
import numpy as np
from typing import Dict, Tuple, Optional

class DefectClassifier(nn.Module):
    """
    CNN-based defect classifier with multiple backbone options.
    Supports EfficientNet, ResNet, and custom architectures.
    """
    
    def __init__(self, 
                 num_classes: int = 7,
                 backbone: str = "efficientnet-b0",
                 pretrained: bool = True,
                 dropout_rate: float = 0.2):
        """
        Initialize the defect classifier.
        
        Args:
            num_classes: Number of defect classes
            backbone: Model backbone ('efficientnet-b0', 'resnet50', 'resnet18')
            pretrained: Use pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super(DefectClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.dropout_rate = dropout_rate
        
      
        self._build_backbone(pretrained)
        
  
        self._build_classifier()
        
    def _build_backbone(self, pretrained: bool):
        """Build the feature extraction backbone."""
        
        if self.backbone_name.startswith("efficientnet"):
            if pretrained:
                self.backbone = EfficientNet.from_pretrained(self.backbone_name)
            else:
                self.backbone = EfficientNet.from_name(self.backbone_name)
            
           
            self.feature_dim = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()
            
        elif self.backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif self.backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
    
    def _build_classifier(self):
        """Build the classification head."""
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        
    
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        features = self.backbone(x)
        
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        return features
    
    def predict(self, x: torch.Tensor, return_probs: bool = True) -> Dict:
        """
        Make predictions on input tensor.
        
        Args:
            x: Input tensor
            return_probs: Whether to return probabilities
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
            
            results = {
                'logits': logits,
                'predicted_classes': predicted_classes,
                'confidence_scores': confidence_scores
            }
            
            if return_probs:
                results['probabilities'] = probabilities
                
            return results
    
    def get_target_layer(self) -> nn.Module:
        """
        Get the target layer for Grad-CAM visualization.
        
        Returns:
            Target layer module
        """
        if self.backbone_name.startswith("efficientnet"):
           
            return self.backbone._blocks[-1]
        elif self.backbone_name.startswith("resnet"):
          
            return self.backbone.layer4[-1]
        else:
            raise ValueError(f"Target layer not defined for {self.backbone_name}")

class DefectClassifierTrainer:
    """Trainer class for the defect classifier."""
    
    def __init__(self, 
                 model: DefectClassifier,
                 device: torch.device,
                 class_names: Optional[Dict[int, str]] = None):
        """
        Initialize trainer.
        
        Args:
            model: DefectClassifier model
            device: Training device
            class_names: Mapping of class indices to names
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names or {
            0: "Normal", 1: "Scratch", 2: "Dent", 3: "Crack",
            4: "Corrosion", 5: "Missing_Part", 6: "Color_Defect"
        }
        
      
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
    def setup_optimizer(self, 
                       learning_rate: float = 0.001,
                       weight_decay: float = 1e-4,
                       optimizer_type: str = "adam"):
        """Setup optimizer and scheduler."""
        
        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            
          
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
           
            loss.backward()
            self.optimizer.step()
            
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
    
    def validate_epoch(self, dataloader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
    
    def save_checkpoint(self, filepath: str, epoch: int, best_accuracy: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': best_accuracy,
            'class_names': self.class_names,
            'model_config': {
                'num_classes': self.model.num_classes,
                'backbone': self.model.backbone_name,
                'dropout_rate': self.model.dropout_rate
            }
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['best_accuracy']

def create_model(config: Dict) -> DefectClassifier:
    """
    Factory function to create defect classifier from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DefectClassifier instance
    """
    model_config = config.get('model', {})
    
    model = DefectClassifier(
        num_classes=model_config.get('num_classes', 7),
        backbone=model_config.get('name', 'efficientnet-b0'),
        pretrained=model_config.get('pretrained', True),
        dropout_rate=0.2
    )
    
    return model
