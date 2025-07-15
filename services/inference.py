import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, Union
import logging
import time

from models.defect_classifier import DefectClassifier, create_model
from models.explainer import DefectExplainer, create_explainer
from services.ocr_service import OCRService, create_ocr_service
from services.llm_service import LLMService, create_llm_service
from utils.image_processing import preprocess_image, resize_image
from utils.config import load_config

class InferencePipeline:
    """
    Main inference pipeline that orchestrates the complete defect analysis workflow:
    1. Image preprocessing
    2. CNN defect classification
    3. Grad-CAM visualization
    4. OCR text extraction
    5. LLM repair suggestions
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the inference pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
        self.class_names = config.get('defect_classes', {
            0: "Normal", 1: "Scratch", 2: "Dent", 3: "Crack",
            4: "Corrosion", 5: "Missing_Part", 6: "Color_Defect"
        })
        
       
        self.classifier = None
        self.explainer = None
        self.ocr_service = None
        self.llm_service = None
        
        
        self._initialize_services()
        
        logging.info("Inference pipeline initialized successfully")
    
    def _initialize_services(self):
        """Initialize all pipeline services."""
        try:
            # Initialize CNN classifier
            self._initialize_classifier()
            
            # Initialize Grad-CAM explainer
            self._initialize_explainer()
            
            # Initialize OCR service
            self._initialize_ocr()
            
            # Initialize LLM service
            self._initialize_llm()
            
        except Exception as e:
            logging.error(f"Error initializing services: {e}")
            raise
    
    def _initialize_classifier(self):
        """Initialize the defect classifier."""
        try:
            logging.info("Initializing defect classifier...")
            
            
            self.classifier = create_model(self.config)
            self.classifier = self.classifier.to(self.device)
            
           
            checkpoint_path = self.config.get('model', {}).get('checkpoint_path')
            if checkpoint_path:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        self.classifier.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.classifier.load_state_dict(checkpoint)
                    logging.info(f"Loaded model weights from {checkpoint_path}")
                except Exception as e:
                    logging.warning(f"Could not load model weights: {e}. Using pretrained backbone only.")
            
            self.classifier.eval()
            logging.info("Defect classifier initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize classifier: {e}")
            raise
    
    def _initialize_explainer(self):
        """Initialize the Grad-CAM explainer."""
        try:
            logging.info("Initializing Grad-CAM explainer...")
            
            self.explainer = create_explainer(
                model=self.classifier,
                class_names=self.class_names,
                device=self.device,
                explainer_type="gradcam"
            )
            
            logging.info("Grad-CAM explainer initialized successfully")
            
        except Exception as e:
            logging.warning(f"Failed to initialize explainer: {e}")
            self.explainer = None
    
    def _initialize_ocr(self):
        """Initialize the OCR service."""
        try:
            logging.info("Initializing OCR service...")
            
            self.ocr_service = create_ocr_service(self.config)
            
            if self.ocr_service.is_available():
                logging.info("OCR service initialized successfully")
            else:
                logging.warning("OCR service not available")
                
        except Exception as e:
            logging.warning(f"Failed to initialize OCR service: {e}")
            self.ocr_service = None
    
    def _initialize_llm(self):
        """Initialize the LLM service."""
        try:
            logging.info("Initializing LLM service...")
            
            self.llm_service = create_llm_service(self.config)
            
            if self.llm_service.is_available():
                logging.info("LLM service initialized successfully")
            else:
                logging.warning("LLM service not available")
                
        except Exception as e:
            logging.warning(f"Failed to initialize LLM service: {e}")
            self.llm_service = None
    
    def analyze_image(self, 
                     image: Union[Image.Image, np.ndarray, str],
                     enable_ocr: bool = True,
                     enable_llm: bool = True,
                     confidence_threshold: float = 0.5,
                     gradcam_alpha: float = 0.4,
                     llm_temperature: float = 0.7) -> Dict:
        """
        Perform complete defect analysis on an image.
        
        Args:
            image: Input image (PIL Image, numpy array, or file path)
            enable_ocr: Whether to perform OCR
            enable_llm: Whether to generate LLM suggestions
            confidence_threshold: Minimum confidence threshold
            gradcam_alpha: Grad-CAM overlay transparency
            llm_temperature: LLM generation temperature
            
        Returns:
            Dictionary containing complete analysis results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            preprocessed_image, original_image = self._preprocess_input(image)
            
            # Step 1: CNN Classification
            classification_results = self._classify_defect(
                preprocessed_image, confidence_threshold
            )
            
            # Step 2: Grad-CAM Visualization
            gradcam_results = self._generate_gradcam(
                preprocessed_image, 
                classification_results['predicted_class'],
                gradcam_alpha
            )
            
            # Step 3: OCR Text Extraction
            ocr_results = self._extract_text(original_image, enable_ocr)
            
            # Step 4: LLM Repair Suggestions
            llm_results = self._generate_suggestions(
                classification_results,
                ocr_results,
                enable_llm,
                llm_temperature
            )
            
            # Compile results
            analysis_time = time.time() - start_time
            
            results = {
                'classification': classification_results,
                'gradcam': gradcam_results,
                'ocr': ocr_results,
                'llm': llm_results,
                'metadata': {
                    'analysis_time': analysis_time,
                    'image_size': original_image.size,
                    'device_used': str(self.device),
                    'services_enabled': {
                        'ocr': enable_ocr and self.ocr_service is not None,
                        'llm': enable_llm and self.llm_service is not None,
                        'gradcam': self.explainer is not None
                    }
                }
            }
            
            logging.info(f"Analysis completed in {analysis_time:.2f} seconds")
            return results
            
        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _preprocess_input(self, image: Union[Image.Image, np.ndarray, str]) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocess input image for analysis.
        
        Args:
            image: Input image in various formats
            
        Returns:
            Tuple of (preprocessed_tensor, original_pil_image)
        """
        # Convert to PIL Image if needed
        if isinstance(image, str):
            original_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            original_image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            original_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Resize image if too large
        max_size = 1024
        if max(original_image.size) > max_size:
            original_image = resize_image(original_image, max_size)
        
        # Preprocess for CNN
        preprocessed_tensor = preprocess_image(
            original_image, 
            target_size=self.config.get('training', {}).get('image_size', 224)
        )
        
        return preprocessed_tensor, original_image
    
    def _classify_defect(self, image_tensor: torch.Tensor, confidence_threshold: float) -> Dict:
        """
        Classify defects using CNN.
        
        Args:
            image_tensor: Preprocessed image tensor
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Classification results dictionary
        """
        try:
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            
            with torch.no_grad():
                self.classifier.eval()
                outputs = self.classifier(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities.max().item()
                
                # Create probability distribution
                prob_dict = {}
                for i, prob in enumerate(probabilities[0]):
                    class_name = self.class_names.get(i, f"Class_{i}")
                    prob_dict[class_name] = float(prob)
                
                return {
                    'predicted_class': predicted_class,
                    'class_name': self.class_names.get(predicted_class, f"Class_{predicted_class}"),
                    'confidence': confidence,
                    'meets_threshold': confidence >= confidence_threshold,
                    'probabilities': prob_dict,
                    'raw_logits': outputs[0].cpu().numpy().tolist()
                }
                
        except Exception as e:
            logging.error(f"Classification failed: {e}")
            return {
                'predicted_class': 0,
                'class_name': 'Normal',
                'confidence': 0.0,
                'meets_threshold': False,
                'probabilities': {},
                'error': str(e)
            }
    
    def _generate_gradcam(self, 
                         image_tensor: torch.Tensor,
                         target_class: int,
                         alpha: float) -> Dict:
        """
        Generate Grad-CAM visualization.
        
        Args:
            image_tensor: Input image tensor
            target_class: Target class for visualization
            alpha: Overlay transparency
            
        Returns:
            Grad-CAM results dictionary
        """
        if self.explainer is None:
            return {
                'heatmap': None,
                'cam': None,
                'available': False,
                'error': 'Grad-CAM explainer not available'
            }
        
        try:
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            explanation = self.explainer.explain_prediction(
                image_tensor,
                target_class=target_class,
                alpha=alpha,
                return_image=True
            )
            
            return {
                'heatmap': explanation.get('heatmap_image'),
                'cam': explanation.get('cam'),
                'available': True,
                'confidence': explanation.get('confidence', 0.0),
                'predicted_class': explanation.get('predicted_class_name', 'Unknown')
            }
            
        except Exception as e:
            logging.error(f"Grad-CAM generation failed: {e}")
            return {
                'heatmap': None,
                'cam': None,
                'available': False,
                'error': str(e)
            }
    
    def _extract_text(self, image: Image.Image, enable_ocr: bool) -> Dict:
        """
        Extract text using OCR.
        
        Args:
            image: PIL Image
            enable_ocr: Whether OCR is enabled
            
        Returns:
            OCR results dictionary
        """
        if not enable_ocr or self.ocr_service is None:
            return {
                'text': '',
                'words': [],
                'confidence': 0.0,
                'available': False,
                'enabled': enable_ocr
            }
        
        try:
            ocr_results = self.ocr_service.extract_text(image)
            
            return {
                'text': ocr_results.get('text', ''),
                'words': ocr_results.get('words', []),
                'confidence': ocr_results.get('confidence', 0.0),
                'language': ocr_results.get('language', 'unknown'),
                'available': True,
                'enabled': True,
                'success': ocr_results.get('success', False)
            }
            
        except Exception as e:
            logging.error(f"OCR extraction failed: {e}")
            return {
                'text': '',
                'words': [],
                'confidence': 0.0,
                'available': False,
                'enabled': True,
                'error': str(e)
            }
    
    def _generate_suggestions(self, 
                            classification_results: Dict,
                            ocr_results: Dict,
                            enable_llm: bool,
                            temperature: float) -> Dict:
        """
        Generate repair suggestions using LLM.
        
        Args:
            classification_results: Classification results
            ocr_results: OCR results
            enable_llm: Whether LLM is enabled
            temperature: Generation temperature
            
        Returns:
            LLM results dictionary
        """
        if not enable_llm or self.llm_service is None:
            return {
                'suggestion': '',
                'template_suggestions': [],
                'available': False,
                'enabled': enable_llm
            }
        
        try:
            # Update temperature if needed
            if hasattr(self.llm_service, 'temperature'):
                self.llm_service.temperature = temperature
            
            suggestions = self.llm_service.generate_repair_suggestion(
                defect_type=classification_results.get('class_name', 'Unknown'),
                confidence=classification_results.get('confidence', 0.0),
                ocr_text=ocr_results.get('text', ''),
                additional_context=""
            )
            
            # Format combined suggestion
            combined_suggestion = self._format_suggestions(suggestions)
            
            return {
                'suggestion': combined_suggestion,
                'template_suggestions': suggestions.get('template_suggestions', []),
                'ai_suggestions': suggestions.get('ai_suggestions', ''),
                'available': True,
                'enabled': True,
                'defect_context': {
                    'type': suggestions.get('defect_type', ''),
                    'severity': suggestions.get('severity', ''),
                    'confidence': suggestions.get('confidence', 0.0)
                }
            }
            
        except Exception as e:
            logging.error(f"LLM suggestion generation failed: {e}")
            return {
                'suggestion': 'Unable to generate repair suggestions at this time.',
                'template_suggestions': [],
                'available': False,
                'enabled': True,
                'error': str(e)
            }
    
    def _format_suggestions(self, suggestions: Dict) -> str:
        """Format suggestions into a readable string."""
        try:
            formatted_parts = []
            
            # Add template suggestions
            template_suggestions = suggestions.get('template_suggestions', [])
            if template_suggestions:
                formatted_parts.append("**Recommended Actions:**")
                for i, suggestion in enumerate(template_suggestions[:3], 1):
                    formatted_parts.append(f"{i}. {suggestion}")
            
            # Add AI suggestions if available
            ai_suggestions = suggestions.get('ai_suggestions', '')
            if ai_suggestions and ai_suggestions.strip():
                if formatted_parts:
                    formatted_parts.append("\n**Additional Insights:**")
                formatted_parts.append(ai_suggestions)
            
            # Add severity information
            severity = suggestions.get('severity', '')
            if severity:
                formatted_parts.append(f"\n**Severity Level:** {severity.title()}")
            
            return "\n".join(formatted_parts) if formatted_parts else "No specific suggestions available."
            
        except Exception as e:
            logging.error(f"Error formatting suggestions: {e}")
            return "Error formatting repair suggestions."
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result structure."""
        return {
            'classification': {
                'predicted_class': 0,
                'class_name': 'Unknown',
                'confidence': 0.0,
                'meets_threshold': False,
                'probabilities': {},
                'error': error_message
            },
            'gradcam': {
                'heatmap': None,
                'available': False,
                'error': error_message
            },
            'ocr': {
                'text': '',
                'available': False,
                'error': error_message
            },
            'llm': {
                'suggestion': 'Analysis failed. Please try again.',
                'available': False,
                'error': error_message
            },
            'metadata': {
                'analysis_time': 0.0,
                'error': error_message
            }
        }
    
    def get_service_status(self) -> Dict[str, bool]:
        """Get status of all services."""
        return {
            'classifier': self.classifier is not None,
            'explainer': self.explainer is not None,
            'ocr': self.ocr_service is not None and self.ocr_service.is_available(),
            'llm': self.llm_service is not None and self.llm_service.is_available()
        }
