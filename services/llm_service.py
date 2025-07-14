import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    GPT2LMHeadModel, GPT2Tokenizer,
    pipeline, AutoModel
)
import logging
from typing import Dict, List, Optional, Union
import json
import re

class LLMService:
    """
    Service for generating repair suggestions using Hugging Face Language Models.
    Supports multiple model types and provides repair recommendation templates.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: str = "auto",
                 max_length: int = 512,
                 temperature: float = 0.7):
        """
        Initialize LLM service.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device for inference ('cpu', 'cuda', 'auto')
            max_length: Maximum generation length
            temperature: Generation temperature
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        self._load_model()
        
        # Repair templates
        self.repair_templates = self._initialize_repair_templates()
    
    def _load_model(self):
        """Load the language model and tokenizer."""
        try:
            logging.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left',
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try to load as a text generation pipeline first
            try:
                self.generator = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                logging.info("Model loaded successfully using pipeline")
                
            except Exception as e:
                logging.warning(f"Pipeline loading failed: {e}. Trying direct model loading...")
                
                # Fallback to direct model loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                logging.info("Model loaded successfully using direct loading")
                
        except Exception as e:
            logging.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a simple model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if the primary model fails."""
        try:
            logging.info("Loading fallback model: gpt2")
            self.model_name = "gpt2"
            
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logging.info("Fallback model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load fallback model: {e}")
            self.model = None
            self.tokenizer = None
    
    def _initialize_repair_templates(self) -> Dict[str, Dict]:
        """Initialize repair suggestion templates for different defect types."""
        return {
            "scratch": {
                "severity_low": [
                    "Apply scratch remover compound and buff with microfiber cloth",
                    "Use fine polishing compound to minimize surface scratches",
                    "Clean area and apply protective coating to prevent further scratching"
                ],
                "severity_medium": [
                    "Sand lightly with 2000-grit sandpaper, apply primer and repaint",
                    "Use touch-up paint pen for small scratches, followed by clear coat",
                    "Apply rubbing compound, polish, and protective wax"
                ],
                "severity_high": [
                    "Professional refinishing required - sand, prime, paint, and clear coat",
                    "Replace damaged component if scratch affects structural integrity",
                    "Contact certified technician for proper repair assessment"
                ]
            },
            "dent": {
                "severity_low": [
                    "Use paintless dent removal (PDR) technique with specialized tools",
                    "Apply heat gun and plunger method for shallow dents",
                    "Use dent pulling kit with suction cups for minor dents"
                ],
                "severity_medium": [
                    "Professional PDR service recommended for optimal results",
                    "Body filler application followed by sanding and repainting",
                    "Heat treatment and gradual reshaping with proper tools"
                ],
                "severity_high": [
                    "Panel replacement may be necessary for severe dents",
                    "Professional body shop assessment required",
                    "Structural integrity inspection before repair attempts"
                ]
            },
            "crack": {
                "severity_low": [
                    "Clean crack thoroughly and apply appropriate sealant",
                    "Use crack repair kit specific to material type",
                    "Monitor crack progression and reapply sealant as needed"
                ],
                "severity_medium": [
                    "Drill stop holes at crack ends to prevent propagation",
                    "Apply structural adhesive or welding repair as appropriate",
                    "Reinforce area with backing material if accessible"
                ],
                "severity_high": [
                    "Immediate replacement of cracked component recommended",
                    "Safety inspection required - component may be compromised",
                    "Professional engineering assessment for structural cracks"
                ]
            },
            "corrosion": {
                "severity_low": [
                    "Remove rust with wire brush and apply rust converter",
                    "Clean with rust remover, prime with anti-corrosion primer",
                    "Apply protective coating to prevent future corrosion"
                ],
                "severity_medium": [
                    "Sand to bare metal, apply rust inhibitor and protective paint",
                    "Use electrolytic rust removal for thorough cleaning",
                    "Apply multi-layer protection: primer, paint, and sealant"
                ],
                "severity_high": [
                    "Component replacement required due to structural compromise",
                    "Professional corrosion assessment and treatment needed",
                    "Investigate root cause of corrosion to prevent recurrence"
                ]
            },
            "missing_part": {
                "severity_low": [
                    "Order replacement part using component serial number",
                    "Install new part following manufacturer specifications",
                    "Verify proper fit and function after installation"
                ],
                "severity_medium": [
                    "Source OEM or compatible replacement component",
                    "May require professional installation for proper alignment",
                    "Update documentation and maintenance records"
                ],
                "severity_high": [
                    "Critical component missing - immediate replacement required",
                    "Safety inspection mandatory before return to service",
                    "Investigate cause of component loss to prevent recurrence"
                ]
            },
            "color_defect": {
                "severity_low": [
                    "Light polishing compound to restore color uniformity",
                    "UV protectant application to prevent further fading",
                    "Professional color matching for spot repairs"
                ],
                "severity_medium": [
                    "Partial repainting of affected area with color matching",
                    "Surface preparation and primer application before painting",
                    "Apply multiple thin coats for even color distribution"
                ],
                "severity_high": [
                    "Complete refinishing required for color consistency",
                    "Professional paint shop service recommended",
                    "Consider substrate condition - may require surface preparation"
                ]
            },
            "normal": {
                "general": [
                    "No defects detected - component appears to be in good condition",
                    "Continue regular inspection and maintenance schedule",
                    "Monitor for early signs of wear or deterioration"
                ]
            }
        }
    
    def generate_repair_suggestion(self, 
                                 defect_type: str,
                                 confidence: float,
                                 ocr_text: str = "",
                                 additional_context: str = "") -> Dict[str, str]:
        """
        Generate repair suggestions based on defect classification and context.
        
        Args:
            defect_type: Type of defect detected
            confidence: Confidence score of detection
            ocr_text: Text extracted from OCR
            additional_context: Additional context information
            
        Returns:
            Dictionary containing repair suggestions
        """
        try:
            # Normalize defect type
            defect_key = defect_type.lower().replace(" ", "_")
            
            # Get severity level based on confidence
            severity = self._determine_severity(confidence)
            
            # Get template-based suggestions
            template_suggestions = self._get_template_suggestions(defect_key, severity)
            
            # Generate AI-based suggestions if model is available
            ai_suggestions = ""
            if self.model is not None or self.generator is not None:
                ai_suggestions = self._generate_ai_suggestions(
                    defect_type, confidence, ocr_text, additional_context
                )
            
            # Combine suggestions
            return {
                "template_suggestions": template_suggestions,
                "ai_suggestions": ai_suggestions,
                "defect_type": defect_type,
                "severity": severity,
                "confidence": confidence,
                "ocr_context": ocr_text
            }
            
        except Exception as e:
            logging.error(f"Error generating repair suggestions: {e}")
            return {
                "template_suggestions": ["Contact qualified technician for repair assessment"],
                "ai_suggestions": "Unable to generate AI suggestions at this time.",
                "error": str(e)
            }
    
    def _determine_severity(self, confidence: float) -> str:
        """Determine severity level based on confidence score."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _get_template_suggestions(self, defect_key: str, severity: str) -> List[str]:
        """Get template-based repair suggestions."""
        if defect_key in self.repair_templates:
            defect_templates = self.repair_templates[defect_key]
            
            # Try to get severity-specific suggestions
            severity_key = f"severity_{severity}"
            if severity_key in defect_templates:
                return defect_templates[severity_key]
            
            # Fallback to general suggestions
            if "general" in defect_templates:
                return defect_templates["general"]
                
            # Return first available suggestions
            for key in defect_templates:
                if isinstance(defect_templates[key], list):
                    return defect_templates[key]
        
        # Default suggestions
        return [
            "Inspect the defected area carefully",
            "Consult manufacturer specifications for repair procedures",
            "Consider professional assessment if unsure about repair method"
        ]
    
    def _generate_ai_suggestions(self, 
                               defect_type: str,
                               confidence: float, 
                               ocr_text: str,
                               additional_context: str) -> str:
        """Generate AI-powered repair suggestions."""
        try:
            # Construct prompt
            prompt = self._construct_repair_prompt(
                defect_type, confidence, ocr_text, additional_context
            )
            
            # Generate using pipeline if available
            if self.generator is not None:
                return self._generate_with_pipeline(prompt)
            
            # Generate using direct model
            elif self.model is not None:
                return self._generate_with_model(prompt)
            
            else:
                return "AI model not available for suggestions"
                
        except Exception as e:
            logging.error(f"AI suggestion generation failed: {e}")
            return f"AI suggestion generation encountered an error: {str(e)}"
    
    def _construct_repair_prompt(self, 
                               defect_type: str,
                               confidence: float,
                               ocr_text: str,
                               additional_context: str) -> str:
        """Construct prompt for repair suggestion generation."""
        prompt_parts = [
            f"Defect Analysis Report:",
            f"- Defect Type: {defect_type}",
            f"- Detection Confidence: {confidence:.2%}",
        ]
        
        if ocr_text.strip():
            prompt_parts.append(f"- Extracted Text: {ocr_text}")
        
        if additional_context.strip():
            prompt_parts.append(f"- Additional Context: {additional_context}")
        
        prompt_parts.extend([
            "",
            "Based on this defect analysis, provide specific repair recommendations:",
            "1. Immediate actions needed:",
            "2. Required tools and materials:",
            "3. Step-by-step repair process:",
            "4. Safety considerations:",
            "5. Quality assurance steps:",
            "",
            "Repair Recommendations:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_with_pipeline(self, prompt: str) -> str:
        """Generate suggestions using Hugging Face pipeline."""
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=200,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
                return self._clean_generated_text(generated_text)
            
            return "No suggestions generated"
            
        except Exception as e:
            logging.error(f"Pipeline generation error: {e}")
            return f"Generation error: {str(e)}"
    
    def _generate_with_model(self, prompt: str) -> str:
        """Generate suggestions using direct model inference."""
        try:
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.max_length - 200
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            return self._clean_generated_text(generated_text)
            
        except Exception as e:
            logging.error(f"Model generation error: {e}")
            return f"Generation error: {str(e)}"
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean and format generated text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove incomplete sentences at the end
        sentences = text.split('.')
        if len(sentences) > 1 and len(sentences[-1]) < 10:
            text = '.'.join(sentences[:-1]) + '.'
        
        # Truncate if too long
        if len(text) > 500:
            text = text[:500] + "..."
        
        return text
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.model is not None or self.generator is not None

class MockLLMService(LLMService):
    """Mock LLM service for testing and development."""
    
    def __init__(self):
        """Initialize mock LLM service."""
        self.model_name = "mock-model"
        self.device = "cpu"
        self.repair_templates = self._initialize_repair_templates()
    
    def _generate_ai_suggestions(self, 
                               defect_type: str,
                               confidence: float,
                               ocr_text: str,
                               additional_context: str) -> str:
        """Generate mock AI suggestions."""
        mock_suggestions = {
            "scratch": "Clean the scratched area thoroughly. Apply touch-up paint or polishing compound based on scratch depth. For deep scratches, consider professional refinishing.",
            "dent": "Assess dent severity. Small dents can be repaired using paintless dent removal techniques. Larger dents may require body filler and repainting.",
            "crack": "Clean crack area and apply appropriate sealant immediately. Monitor for crack propagation. Consider component replacement if crack affects structural integrity.",
            "corrosion": "Remove rust using appropriate tools and chemicals. Apply rust inhibitor and protective coating. Address underlying moisture issues to prevent recurrence.",
            "missing_part": "Identify missing component using extracted serial number if available. Order OEM replacement part and follow manufacturer installation guidelines.",
            "color_defect": "Assess color variation extent. Light polishing may restore uniformity. Significant color defects require professional color matching and repainting."
        }
        
        defect_key = defect_type.lower().replace(" ", "_")
        base_suggestion = mock_suggestions.get(defect_key, "Consult manufacturer guidelines for appropriate repair procedures.")
        
        # Add context if OCR text is available
        if ocr_text.strip():
            base_suggestion += f" Note: Extracted text '{ocr_text}' may provide additional component information for sourcing parts."
        
        return base_suggestion
    
    def is_available(self) -> bool:
        """Mock service is always available."""
        return True

def create_llm_service(config: Dict) -> LLMService:
    """
    Factory function to create LLM service from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LLMService instance
    """
    llm_config = config.get('llm', {})
    
    # Check if running in development mode
    if config.get('development', {}).get('use_mock_llm', False):
        return MockLLMService()
    
    return LLMService(
        model_name=llm_config.get('model_name', 'microsoft/DialoGPT-medium'),
        device=llm_config.get('device', 'auto'),
        max_length=llm_config.get('max_length', 512),
        temperature=llm_config.get('temperature', 0.7)
    )
