import os
import io
from typing import Dict, List, Optional, Tuple
from PIL import Image
import logging

try:
    from google.cloud import vision
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    logging.warning("Google Cloud Vision API not available. Install with: pip install google-cloud-vision")

class OCRService:
    """
    Service for extracting text from images using Google Cloud Vision API.
    Provides fallback functionality and error handling.
    """
    
    def __init__(self, credentials_path: Optional[str] = None, project_id: Optional[str] = None):
        """
        Initialize OCR service.
        
        Args:
            credentials_path: Path to Google Cloud service account key
            project_id: Google Cloud project ID
        """
        self.credentials_path = credentials_path
        self.project_id = project_id
        self.client = None
        
        if GOOGLE_CLOUD_AVAILABLE:
            self._initialize_client()
        else:
            logging.warning("Google Cloud Vision API not available")
    
    def _initialize_client(self):
        """Initialize Google Cloud Vision client."""
        try:
            # Set credentials if provided
            if self.credentials_path and os.path.exists(self.credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
            
            # Initialize client
            self.client = vision.ImageAnnotatorClient()
            logging.info("Google Cloud Vision API client initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Google Cloud Vision client: {e}")
            self.client = None
    
    def extract_text(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict:
        """
        Extract text from image using OCR.
        
        Args:
            image: PIL Image object
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        if not self.client:
            return self._fallback_ocr(image)
        
        try:
            # Convert PIL Image to bytes
            image_bytes = self._image_to_bytes(image)
            
            # Create Vision API image object
            vision_image = vision.Image(content=image_bytes)
            
            # Detect text
            response = self.client.text_detection(image=vision_image)
            
            # Process results
            return self._process_vision_response(response, confidence_threshold)
            
        except Exception as e:
            logging.error(f"OCR extraction failed: {e}")
            return self._fallback_ocr(image)
    
    def extract_text_with_locations(self, image: Image.Image) -> Dict:
        """
        Extract text with bounding box locations.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing text and location information
        """
        if not self.client:
            return self._fallback_ocr(image)
        
        try:
            image_bytes = self._image_to_bytes(image)
            vision_image = vision.Image(content=image_bytes)
            
            # Use document text detection for better structure
            response = self.client.document_text_detection(image=vision_image)
            
            return self._process_document_response(response)
            
        except Exception as e:
            logging.error(f"OCR with locations failed: {e}")
            return self._fallback_ocr(image)
    
    def extract_handwritten_text(self, image: Image.Image) -> Dict:
        """
        Extract handwritten text from image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing extracted handwritten text
        """
        if not self.client:
            return self._fallback_ocr(image)
        
        try:
            image_bytes = self._image_to_bytes(image)
            vision_image = vision.Image(content=image_bytes)
            
            # Use handwriting OCR feature
            response = self.client.document_text_detection(
                image=vision_image,
                image_context=vision.ImageContext(
                    language_hints=["en"],  # Add more languages as needed
                )
            )
            
            return self._process_vision_response(response)
            
        except Exception as e:
            logging.error(f"Handwritten text extraction failed: {e}")
            return self._fallback_ocr(image)
    
    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes."""
        img_byte_arr = io.BytesIO()
        
        # Convert to RGB if necessary
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        image.save(img_byte_arr, format='JPEG', quality=95)
        return img_byte_arr.getvalue()
    
    def _process_vision_response(self, response, confidence_threshold: float = 0.5) -> Dict:
        """
        Process Google Vision API response.
        
        Args:
            response: Vision API response
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Processed text extraction results
        """
        results = {
            'text': '',
            'words': [],
            'confidence': 0.0,
            'language': 'unknown',
            'success': True,
            'error': None
        }
        
        try:
            # Check for errors
            if response.error.message:
                results['success'] = False
                results['error'] = response.error.message
                return results
            
            # Extract full text
            if response.text_annotations:
                results['text'] = response.text_annotations[0].description.strip()
                
                # Extract individual words with confidence
                for annotation in response.text_annotations[1:]:  # Skip first (full text)
                    word_info = {
                        'text': annotation.description,
                        'confidence': getattr(annotation, 'confidence', 1.0),
                        'bounding_box': self._extract_bounding_box(annotation)
                    }
                    
                    if word_info['confidence'] >= confidence_threshold:
                        results['words'].append(word_info)
                
                # Calculate average confidence
                if results['words']:
                    results['confidence'] = sum(w['confidence'] for w in results['words']) / len(results['words'])
                else:
                    results['confidence'] = 1.0 if results['text'] else 0.0
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing Vision API response: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results
    
    def _process_document_response(self, response) -> Dict:
        """Process document text detection response."""
        results = {
            'text': '',
            'blocks': [],
            'paragraphs': [],
            'words': [],
            'success': True,
            'error': None
        }
        
        try:
            if response.error.message:
                results['success'] = False
                results['error'] = response.error.message
                return results
            
            document = response.full_text_annotation
            if not document:
                return results
            
            results['text'] = document.text
            
            # Extract structured information
            for page in document.pages:
                for block in page.blocks:
                    block_text = ""
                    block_words = []
                    
                    for paragraph in block.paragraphs:
                        paragraph_text = ""
                        paragraph_words = []
                        
                        for word in paragraph.words:
                            word_text = ''.join([symbol.text for symbol in word.symbols])
                            word_info = {
                                'text': word_text,
                                'confidence': word.confidence,
                                'bounding_box': self._extract_bounding_box(word)
                            }
                            
                            paragraph_text += word_text + " "
                            paragraph_words.append(word_info)
                            block_words.append(word_info)
                        
                        results['paragraphs'].append({
                            'text': paragraph_text.strip(),
                            'words': paragraph_words,
                            'bounding_box': self._extract_bounding_box(paragraph)
                        })
                        
                        block_text += paragraph_text
                    
                    results['blocks'].append({
                        'text': block_text.strip(),
                        'words': block_words,
                        'bounding_box': self._extract_bounding_box(block)
                    })
            
            # Flatten words for compatibility
            for block in results['blocks']:
                results['words'].extend(block['words'])
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing document response: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results
    
    def _extract_bounding_box(self, annotation) -> Dict:
        """Extract bounding box coordinates from annotation."""
        if hasattr(annotation, 'bounding_poly') and annotation.bounding_poly:
            vertices = annotation.bounding_poly.vertices
            return {
                'x1': min(v.x for v in vertices),
                'y1': min(v.y for v in vertices),
                'x2': max(v.x for v in vertices),
                'y2': max(v.y for v in vertices)
            }
        return {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
    
    def _fallback_ocr(self, image: Image.Image) -> Dict:
        """
        Fallback OCR implementation when Google Cloud Vision is not available.
        This is a basic implementation - consider integrating Tesseract or EasyOCR.
        """
        logging.warning("Using fallback OCR (basic implementation)")
        
        return {
            'text': '',
            'words': [],
            'confidence': 0.0,
            'language': 'unknown',
            'success': False,
            'error': 'Google Cloud Vision API not available. Please install and configure it.'
        }
    
    def is_available(self) -> bool:
        """Check if OCR service is available."""
        return self.client is not None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # This is a subset of languages supported by Google Cloud Vision
        return [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
            'ar', 'hi', 'th', 'vi', 'nl', 'sv', 'da', 'no', 'fi', 'pl'
        ]

class MockOCRService(OCRService):
    """
    Mock OCR service for testing and development.
    Provides simulated OCR responses.
    """
    
    def __init__(self):
        """Initialize mock OCR service."""
        super().__init__(credentials_path=None, project_id=None)
        self.client = "mock"  # Set to non-None to bypass availability check
    
    def extract_text(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict:
        """Mock text extraction."""
        # Simulate OCR results based on image characteristics
        width, height = image.size
        
        # Generate mock text based on image properties
        mock_texts = [
            "SERIAL: ABC123456",
            "MODEL: XYZ-2024",
            "BATCH: 20240714",
            "PART#: DEF789",
            "QC PASSED",
            "DEFECT CODE: D001"
        ]
        
        # Select mock text based on image hash
        text_index = hash(str(image.tobytes())) % len(mock_texts)
        selected_text = mock_texts[text_index]
        
        return {
            'text': selected_text,
            'words': [
                {
                    'text': word,
                    'confidence': 0.85 + (hash(word) % 15) / 100,  # 0.85-1.00
                    'bounding_box': {
                        'x1': 10, 'y1': 10, 
                        'x2': width-10, 'y2': height-10
                    }
                }
                for word in selected_text.split()
            ],
            'confidence': 0.92,
            'language': 'en',
            'success': True,
            'error': None
        }

def create_ocr_service(config: Dict) -> OCRService:
    """
    Factory function to create OCR service from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        OCRService instance
    """
    google_config = config.get('google_cloud', {})
    
    # Check if running in development mode
    if config.get('development', {}).get('use_mock_ocr', False):
        return MockOCRService()
    
    return OCRService(
        credentials_path=google_config.get('credentials_path'),
        project_id=google_config.get('project_id')
    )
