import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.inference import InferencePipeline
from utils.config import load_config
from utils.image_processing import preprocess_image, resize_image

# Page configuration
st.set_page_config(
    page_title="Faultiz - Visual Defect Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .defect-detected {
        color: #d62728;
        font-weight: bold;
    }
    .normal-detected {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_inference_pipeline():
    """Load the inference pipeline with caching."""
    config = load_config()
    return InferencePipeline(config)

def display_results(results, uploaded_image):
    """Display analysis results in organized columns."""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Display metrics
        st.subheader("📊 Classification Results")
        confidence = results['classification']['confidence']
        predicted_class = results['classification']['class_name']
        
        if predicted_class == "Normal":
            st.markdown(f"<div class='normal-detected'>✅ Status: {predicted_class}</div>", 
                       unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='defect-detected'>⚠️ Defect Detected: {predicted_class}</div>", 
                       unsafe_allow_html=True)
        
        st.metric("Confidence", f"{confidence:.2%}")
        
        # Confidence distribution
        if 'probabilities' in results['classification']:
            st.subheader("📈 Confidence Distribution")
            probs = results['classification']['probabilities']
            classes = list(probs.keys())
            values = list(probs.values())
            
            fig = px.bar(x=classes, y=values, 
                        title="Class Probabilities",
                        color=values,
                        color_continuous_scale="viridis")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Grad-CAM Heatmap")
        if results['gradcam']['heatmap'] is not None:
            st.image(results['gradcam']['heatmap'], 
                    caption="Defect Localization Heatmap", 
                    use_column_width=True)
        else:
            st.warning("Grad-CAM visualization not available")
        
        # OCR Results
        st.subheader("📝 Text Extraction (OCR)")
        if results['ocr']['text']:
            st.text_area("Extracted Text:", 
                        value=results['ocr']['text'], 
                        height=100)
        else:
            st.info("No text detected in the image")
    
    # LLM Suggestions (Full Width)
    st.subheader("🤖 AI Repair Suggestions")
    if results['llm']['suggestion']:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>💡 Recommended Actions:</h4>
            <p>{results['llm']['suggestion']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No specific repair suggestions available")

def main():
    # Header
    st.markdown("<h1 class='main-header'>🔍 Faultiz - Visual Defect Classifier</h1>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            AI-powered defect detection with explainable AI and repair recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model:",
            ["EfficientNet-B0", "ResNet50", "Custom"],
            help="Choose the CNN backbone for defect classification"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for defect detection"
        )
        
        # Grad-CAM settings
        st.subheader("🔥 Grad-CAM Settings")
        gradcam_alpha = st.slider(
            "Heatmap Opacity:",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.1
        )
        
        # OCR settings
        st.subheader("📝 OCR Settings")
        enable_ocr = st.checkbox("Enable OCR", value=True)
        
        # LLM settings
        st.subheader("🤖 LLM Settings")
        enable_llm = st.checkbox("Enable Repair Suggestions", value=True)
        llm_temperature = st.slider(
            "Creativity Level:",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
    
    # Main content area
    st.header("📤 Upload Image for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to analyze for defects"
    )
    
    if uploaded_file is not None:
        # Display upload success
        st.success("✅ Image uploaded successfully!")
        
        # Load image
        image = Image.open(uploaded_file)
        
        # Create analysis button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button(
                "🔍 Analyze Image", 
                type="primary",
                use_container_width=True
            )
        
        if analyze_button:
            try:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load pipeline
                status_text.text("Loading AI models...")
                progress_bar.progress(20)
                pipeline = load_inference_pipeline()
                
                # Run inference
                status_text.text("Analyzing image...")
                progress_bar.progress(40)
                
                results = pipeline.analyze_image(
                    image,
                    enable_ocr=enable_ocr,
                    enable_llm=enable_llm,
                    confidence_threshold=confidence_threshold,
                    gradcam_alpha=gradcam_alpha,
                    llm_temperature=llm_temperature
                )
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.divider()
                st.header("📋 Analysis Results")
                display_results(results, image)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
    
    else:
        # Show sample instructions
        st.info("""
        📋 **Instructions:**
        1. Upload an image using the file uploader above
        2. Adjust settings in the sidebar if needed
        3. Click "Analyze Image" to start the analysis
        4. View results including defect classification, heatmaps, OCR text, and repair suggestions
        """)
        
        # Show sample defect types
        st.subheader("🏷️ Supported Defect Types")
        defect_types = [
            "✅ Normal (No Defects)",
            "🔍 Scratches",
            "🔨 Dents", 
            "💥 Cracks",
            "🦠 Corrosion",
            "❌ Missing Parts",
            "🎨 Color Defects"
        ]
        
        cols = st.columns(4)
        for i, defect_type in enumerate(defect_types):
            with cols[i % 4]:
                st.markdown(f"- {defect_type}")

if __name__ == "__main__":
    main()
