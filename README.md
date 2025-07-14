# Faultiz - Visual Defect Classifier & Vision LLM Assistant

A comprehensive AI-powered solution for industrial defect detection, analysis, and repair recommendations.

## 🚀 Features

- **CNN-based Defect Classification**: ResNet/EfficientNet backbone trained on industrial defect datasets
- **Grad-CAM Explainability**: Visual heatmaps showing defect locations
- **Google Cloud Vision OCR**: Text extraction from defective items (serial numbers, labels)
- **Hugging Face LLM Integration**: AI-powered repair suggestions
- **Streamlit UI**: User-friendly interface for complete workflow

## 🏗️ Architecture

```
Image → CNN Classifier → Grad-CAM → Defect Label
   ↓
OCR API → Text Extraction
   ↓
Defect + OCR → LLM → Repair Suggestions
   ↓
Streamlit Dashboard
```

## 📁 Project Structure

```
faultiz/
├── app/
│   ├── __init__.py
│   ├── main.py              # Streamlit main app
│   └── components/          # UI components
├── models/
│   ├── __init__.py
│   ├── defect_classifier.py # CNN model
│   ├── explainer.py         # Grad-CAM implementation
│   └── pretrained/          # Model weights
├── services/
│   ├── __init__.py
│   ├── ocr_service.py       # Google Cloud Vision
│   ├── llm_service.py       # Hugging Face LLM
│   └── inference.py         # Main inference pipeline
├── utils/
│   ├── __init__.py
│   ├── image_processing.py
│   ├── config.py
│   └── data_loader.py
├── data/
│   ├── sample_images/
│   └── defect_classes.json
├── requirements.txt
├── config.yaml
└── setup.py
```

## 🛠️ Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Google Cloud credentials
4. Run the app: `streamlit run app/main.py`

## 🔧 Configuration

- Update `config.yaml` with your API keys and model paths
- Place Google Cloud service account key in `credentials/`

## 📊 Supported Defect Types

- Scratches
- Dents
- Cracks
- Corrosion
- Missing Parts
- Color Defects
- Shape Anomalies

## 🤖 Models Used

- **CNN Backbone**: EfficientNet-B0 (customizable)
- **LLM**: Hugging Face Transformers (configurable model)
- **OCR**: Google Cloud Vision API

## 📝 License

MIT License
