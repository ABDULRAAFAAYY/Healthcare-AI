# Healthcare AI - Intelligent Disease Prediction System

A cutting-edge full-stack web application that leverages **Artificial Intelligence** to assist in early disease detection through two primary modules: **Symptom-Based Disease Prediction** and **Pneumonia Image Detection** using chest X-ray analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation Guide](#installation-guide)
- [Usage Instructions](#usage-instructions)
- [AI Models Documentation](#ai-models-documentation)
- [API Reference](#api-reference)
- [Frontend Components](#frontend-components)
- [Training Custom Models](#training-custom-models)
- [Troubleshooting](#troubleshooting)
- [Project Files Explained](#project-files-explained)
- [Medical Disclaimer](#medical-disclaimer)
- [Credits](#credits)

---

## 🎯 Overview

Healthcare AI is an educational full-stack application designed to demonstrate the power of AI in medical diagnosis. The system combines:

1. **Natural Language Processing (NLP)** for symptom analysis
2. **Computer Vision (Vision Transformers)** for medical imaging analysis

The application provides real-time diagnostic support with:
- Confidence scores for each prediction
- Severity assessments
- Detailed medical recommendations
- Educational information about detected conditions

---

## 🌟 Key Features

### Pneumonia Image Detector
- **AI Model**: Vision Transformer (ViT) from Hugging Face
- **Model Name**: `nickmuchi/vit-finetuned-chest-xray-pneumonia`
- **Input**: Chest X-ray images (JPG, PNG, JPEG)
- **Output**: Binary classification (Normal vs. Pneumonia)
- **Features**:
  - Drag-and-drop image upload
  - Real-time AI analysis
  - Confidence scores for each prediction
  - Severity assessment
  - Medical recommendations
  - Image preview before analysis

### Symptom-Based Disease Predictor
- **AI Logic**: Probabilistic mapping engine
- **Input**: Natural language symptoms (searchable interface)
- **Output**: Top 3 disease predictions with likelihood percentages
- **Features**:
  - Searchable symptom selector with 100+ symptoms
  - Multi-symptom selection
  - Detailed disease descriptions
  - Precaution recommendations
  - Specialist consultation suggestions

### User Interface
- **Modern Design**: Premium dark theme with gradient accents
- **Responsive**: Works seamlessly on desktop, tablet, and mobile devices
- **Interactive**: Smooth animations and transitions
- **Accessible**: ARIA labels and keyboard navigation support
- **User-Friendly**: Intuitive workflows with clear feedback

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  React Frontend │
│   (Port 3000)   │
└────────┬────────┘
         │
         │ HTTP/REST API
         ▼
┌─────────────────┐
│   Flask Backend │
│   (Port 5000)   │
└────────┬────────┘
         │
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│ Hugging Face │ │   Symptom   │ │    Keras     │
│ ViT Model    │ │   Predictor │ │ (Fallback)   │
│ (PyTorch)    │ │  (sklearn)  │ │              │
└──────────────┘ └─────────────┘ └──────────────┘
```

**Flow**:
1. User interacts with React frontend
2. Frontend sends API requests to Flask backend
3. Backend processes requests and routes to appropriate AI model
4. Models return predictions
5. Backend formats and sends response to frontend
6. Frontend displays results with rich visualizations

---

## 💻 Technology Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Core programming language |
| **Flask** | 3.0.0 | Web framework for REST API |
| **Flask-CORS** | 4.0.0 | Cross-origin resource sharing |
| **PyTorch** | 2.1.0 | Deep learning framework |
| **Transformers** | 4.35.0 | Hugging Face library for ViT model |
| **TensorFlow** | 2.15.0 | Alternative deep learning framework |
| **scikit-learn** | 1.3.0 | Machine learning utilities |
| **Pandas** | 2.0.3 | Data manipulation |
| **NumPy** | 1.24.3 | Numerical computing |
| **Pillow** | 10.1.0 | Image processing |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.2.0 | UI library |
| **React Router** | 6.x | Client-side routing |
| **Axios** | Latest | HTTP client for API calls |
| **Vite** | 5.x | Build tool and dev server |
| **React Icons** | Latest | Icon library |
| **CSS3** | - | Styling with modern features |

---

## 📁 Project Structure

```
Healthcare-AI/
│
├── backend/                          # Flask API Backend
│   ├── app.py                       # Main Flask application entry point
│   ├── requirements.txt             # Python dependencies
│   │
│   ├── ml/                          # Machine Learning modules
│   │   ├── __init__.py             # Package initializer
│   │   ├── model_loader.py         # Model loading utilities
│   │   ├── predictor.py            # Symptom-based prediction logic
│   │   ├── image_predictor.py      # Image analysis orchestrator
│   │   └── huggingface_xray_predictor.py  # HF ViT model handler
│   │
│   └── models/                      # Trained ML models storage
│       ├── symptom_model.pkl       # Symptom prediction model
│       ├── label_encoder.pkl       # Label encoder for diseases
│       ├── tfidf_vectorizer.pkl    # Text vectorizer for symptoms
│       └── xray_model.h5           # Keras X-ray model (optional)
│
├── frontend/                         # React Frontend Application
│   ├── public/                      # Static assets
│   │   └── vite.svg
│   │
│   ├── src/
│   │   ├── components/             # Reusable React components
│   │   │   ├── Header.jsx          # Navigation header
│   │   │   ├── Footer.jsx          # Page footer
│   │   │   ├── ImageUpload.jsx     # X-ray image upload component
│   │   │   ├── ResultCard.jsx      # Prediction results display
│   │   │   ├── SymptomSelector.jsx # Symptom selection interface
│   │   │   └── *.css               # Component-specific styles
│   │   │
│   │   ├── pages/                  # Page components (routes)
│   │   │   ├── Home.jsx            # Landing page
│   │   │   ├── DiseasePredictor.jsx # Symptom prediction page
│   │   │   ├── ImagePredictor.jsx  # Pneumonia detection page
│   │   │   ├── About.jsx           # About page
│   │   │   └── *.css               # Page-specific styles
│   │   │
│   │   ├── utils/
│   │   │   └── api.js              # API client functions
│   │   │
│   │   ├── App.jsx                 # Main React component
│   │   ├── main.jsx                # React entry point
│   │   └── index.css               # Global styles and CSS variables
│   │
│   ├── package.json                # Node dependencies
│   ├── vite.config.js              # Vite configuration
│   └── index.html                  # HTML template
│
├── scripts/                          # Training and utility scripts
│   ├── data_preprocessing.py       # Data cleaning and preparation
│   ├── train_symptom_model.py      # Train symptom prediction model
│   ├── train_xray_model.py         # Train X-ray CNN model
│   └── generate_sample_data.py     # Generate demo datasets
│
├── data/                            # Datasets for training
│   ├── symptoms_dataset.csv        # Symptom-disease mappings
│   └── xray_dataset/               # X-ray images (not included)
│       ├── normal/
│       └── pneumonia/
│
├── test_huggingface_model.py       # Test HF model integration
├── test_image_prediction.py        # Test image prediction pipeline
├── setup_xray_model.py             # Setup script for X-ray model
├── project_penaflex.html           # Project poster (6x4 ft)
│
├── QUICKSTART.md                   # Quick start guide
├── TRAIN_MODELS.md                 # Model training guide
├── HUGGINGFACE_MODEL_GUIDE.md      # HF integration documentation
├── IMAGE_PREDICTION_FIX.md         # Troubleshooting guide
└── README.md                       # This file
```

---

## 🚀 Installation Guide

### Prerequisites

Ensure you have the following installed:
- **Python**: 3.11 or higher ([Download](https://www.python.org/downloads/))
- **Node.js**: 16.x or higher ([Download](https://nodejs.org/))
- **npm**: Comes with Node.js
- **Git**: For cloning the repository (optional)

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd Healthcare-AI

# Or download and extract the ZIP file
```

### Step 2: Backend Setup

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create a virtual environment** (highly recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install all required packages including:
   - Flask, PyTorch, Transformers, TensorFlow, scikit-learn, etc.

4. **First-time setup** (Download Hugging Face model):
   ```bash
   python ../test_huggingface_model.py
   ```
   
   This will:
   - Download the Vision Transformer model (~500MB)
   - Cache it locally for future use
   - Verify the installation

5. **Start the Flask backend**:
   ```bash
   python app.py
   ```
   
   You should see:
   ```
   ============================================================
   Healthcare AI API Server
   ============================================================
   Symptom Predictor Status: Ready
   Image Predictor Status: Ready
   ============================================================
   Server starting on http://localhost:5000
   ============================================================
   ```

### Step 3: Frontend Setup

1. **Open a new terminal** (keep backend running)

2. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

3. **Install Node dependencies**:
   ```bash
   npm install
   ```
   
   This will install React, Vite, and all required packages.

4. **Start the development server**:
   ```bash
   npm run dev
   ```
   
   The application will open at `http://localhost:3000`

5. **Access the application**:
   - Open your browser and go to `http://localhost:3000`
   - You should see the Healthcare AI homepage

---

## 📖 Usage Instructions

### Using the Pneumonia Image Detector

1. **Navigate** to "Pneumonia Image Detector" from the main menu
2. **Upload an X-ray image**:
   - Click the upload area or drag and drop an image
   - Supported formats: JPG, PNG, JPEG
   - Maximum file size: 10MB
3. **Click "Analyze Image"**
4. **View results**:
   - Classification: Normal or Pneumonia
   - Confidence score (e.g., 87.3%)
   - Severity level
   - Medical recommendations

**Sample X-ray images** can be found at:
- [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [NIH Chest X-Ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)

### Using the Symptom-Based Predictor

1. **Navigate** to "Disease Predictor" from the main menu
2. **Select symptoms**:
   - Type in the search box to find symptoms
   - Click to select multiple symptoms
   - Selected symptoms appear as tags
3. **Click "Analyze Symptoms"**
4. **View results**:
   - Top 3 disease predictions
   - Probability percentages
   - Detailed descriptions
   - Recommended precautions
   - Specialist suggestions

**Example symptoms to try**:
- Fever, Cough, Fatigue → Likely respiratory infection
- Headache, Nausea, Dizziness → Possible conditions related to these

---

## 🤖 AI Models Documentation

### 1. Pneumonia Image Detector

#### Model Details
- **Model**: Vision Transformer (ViT)
- **Source**: Hugging Face Hub
- **Model ID**: `nickmuchi/vit-finetuned-chest-xray-pneumonia`
- **Architecture**: Vision Transformer (ViT-Base)
- **Training**: Fine-tuned on chest X-ray pneumonia dataset
- **Classes**: 2 (Normal, Pneumonia)

#### How It Works
1. **Image Upload**: User uploads chest X-ray
2. **Preprocessing**:
   - Image resized to 224×224 pixels
   - Converted to RGB if needed
   - Normalized using ViT feature extractor
3. **Inference**:
   - Image passed through Vision Transformer
   - Outputs probabilities for each class
   - Softmax applied for final confidence scores
4. **Post-processing**:
   - Results sorted by confidence
   - Medical information added
   - Formatted response returned

#### Model Performance
- Trained on thousands of chest X-ray images
- Binary classification (Normal vs Pneumonia)
- Provides confidence scores for interpretability

#### File Location
- Handler: `backend/ml/huggingface_xray_predictor.py`
- Model cache: Downloaded to `~/.cache/huggingface/`

### 2. Symptom-Based Disease Predictor

#### Model Details
- **Algorithm**: Rule-based probabilistic mapping
- **Input**: List of symptom strings
- **Output**: Disease probabilities
- **Knowledge Base**: 40+ diseases, 100+ symptoms

#### How It Works
1. **Symptom Input**: User selects symptoms from searchable list
2. **Matching**:
   - Symptoms matched against disease database
   - Each disease has associated symptom patterns
3. **Probability Calculation**:
   - Likelihood computed based on symptom overlap
   - Weighted by symptom severity and specificity
4. **Ranking**:
   - Diseases sorted by probability
   - Top 3 returned with details

#### File Location
- Handler: `backend/ml/predictor.py`
- Model: `backend/models/symptom_model.pkl`

---

## 🔌 API Reference

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. Health Check
```http
GET /api/health
```

**Response**:
```json
{
  "status": "healthy",
  "message": "Healthcare AI API is running",
  "version": "1.0.0"
}
```

#### 2. Get Available Symptoms
```http
GET /api/symptoms/list
```

**Response**:
```json
{
  "symptoms": ["fever", "cough", "fatigue", ...],
  "count": 106
}
```

#### 3. Predict Disease from Symptoms
```http
POST /api/predict/symptoms
Content-Type: application/json
```

**Request Body**:
```json
{
  "symptoms": ["fever", "cough", "fatigue"]
}
```

**Response**:
```json
{
  "success": true,
  "predictions": [
    {
      "disease": "Common Cold",
      "probability": 0.78,
      "description": "Viral infection of upper respiratory tract",
      "precautions": ["Rest", "Stay hydrated", ...],
      "specialist": "General Physician"
    },
    ...
  ]
}
```

#### 4. Predict from Medical Image
```http
POST /api/predict/image
Content-Type: multipart/form-data
```

**Request**:
- Form field: `image` (file)

**Response**:
```json
{
  "success": true,
  "model_type": "huggingface",
  "predictions": [
    {
      "condition": "Normal",
      "confidence": 0.873,
      "severity": "None",
      "description": "No abnormalities detected",
      "recommendations": ["Continue regular check-ups", ...]
    },
    {
      "condition": "Pneumonia",
      "confidence": 0.127,
      "severity": "Moderate to Severe",
      "description": "Inflammation of the lungs",
      "recommendations": ["Consult pulmonologist", ...]
    }
  ],
  "note": "[OK] Using pre-trained Hugging Face model",
  "disclaimer": "[WARNING] For educational purposes only"
}
```

---

## 🎨 Frontend Components

### Core Components

#### `Header.jsx`
- Navigation bar with menu links
- Responsive hamburger menu for mobile
- Active route highlighting

#### `Footer.jsx`
- Footer with copyright information
- Links to social media (placeholder)

#### `ImageUpload.jsx`
- Drag-and-drop file upload interface
- Image preview before analysis
- File validation (type, size)
- Loading states during prediction

#### `ResultCard.jsx`
- Displays prediction results
- Color-coded severity badges
- Confidence bars with animations
- Expandable recommendations

#### `SymptomSelector.jsx`
- Searchable symptom input
- Multi-select with tag display
- Keyboard navigation support

### Pages

#### `Home.jsx`
- Landing page with hero section
- Feature cards for each module
- Call-to-action buttons

#### `DiseasePredictor.jsx`
- Symptom-based prediction interface
- Integrates SymptomSelector
- Displays results with ResultCard

#### `ImagePredictor.jsx`
- X-ray analysis interface
- Integrates ImageUpload
- Shows analysis results

#### `About.jsx`
- Project information
- Technology stack details
- Team credits

---

## 🎓 Training Custom Models

### Symptom Prediction Model

If you want to train your own symptom prediction model:

1. **Prepare dataset**: `data/symptoms_dataset.csv`
   - Format: Columns for symptoms, one column for disease
   - See existing file for example

2. **Run training script**:
   ```bash
   python scripts/train_symptom_model.py
   ```

3. **Output**:
   - `backend/models/symptom_model.pkl`
   - `backend/models/label_encoder.pkl`
   - `backend/models/tfidf_vectorizer.pkl`

### X-Ray Classification Model (Keras)

For training a custom Keras CNN model:

1. **Prepare dataset**: Organize in `data/xray_dataset/`
   ```
   xray_dataset/
   ├── normal/
   │   ├── image1.jpg
   │   └── image2.jpg
   └── pneumonia/
       ├── image1.jpg
       └── image2.jpg
   ```

2. **Run training script**:
   ```bash
   python scripts/train_xray_model.py
   ```

3. **Output**: `backend/models/xray_model.h5`

**Note**: The Hugging Face model is recommended as it's pre-trained and performs well. Custom training is optional.

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Backend won't start**
**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
# Ensure virtual environment is activated
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. **Model loading fails**
**Error**: `Model not loaded. Please check your internet connection`

**Solution**:
- First-time setup requires internet to download model
- Run: `python test_huggingface_model.py`
- Model will be cached locally (~500MB)

#### 3. **TensorFlow conflict error**
**Error**: `AttributeError: module 'tensorflow' has no attribute 'Tensor'`

**Solution**: Already fixed in `huggingface_xray_predictor.py` with monkey-patch. If you still see this:
```bash
# Reinstall transformers
pip uninstall transformers
pip install transformers==4.35.0
```

#### 4. **CORS errors in browser**
**Error**: `Access to XMLHttpRequest blocked by CORS policy`

**Solution**:
- Ensure Flask-CORS is installed: `pip install flask-cors`
- Check that backend is running on port 5000
- Verify frontend is making requests to `http://localhost:5000`

#### 5. **Image upload fails**
**Error**: `Failed to analyze image`

**Solution**:
- Check image format (JPG, PNG, JPEG only)
- Verify file size (max 10MB)
- Ensure image is a valid chest X-ray
- Check backend console for detailed error

### Debugging Tips

**Enable verbose logging**:
```python
# In backend/app.py, set debug=True
app.run(debug=True, host='0.0.0.0', port=5000)
```

**Check backend logs**:
- Console shows request/response info
- Look for error stack traces

**Check browser console**:
- F12 → Console tab
- Look for network errors or API response codes

---

## 📚 Project Files Explained

### Backend Files

| File | Purpose |
|------|---------|
| `app.py` | Flask application entry point, defines API routes |
| `requirements.txt` | Python package dependencies |
| `ml/model_loader.py` | Utilities for loading ML models (Keras, pickle) |
| `ml/predictor.py` | Symptom-based disease prediction logic |
| `ml/image_predictor.py` | Orchestrates image prediction (HF, Keras, demo fallback) |
| `ml/huggingface_xray_predictor.py` | Hugging Face Vision Transformer handler |

### Frontend Files

| File | Purpose |
|------|---------|
| `src/main.jsx` | React application entry point |
| `src/App.jsx` | Main app component with routing |
| `src/index.css` | Global styles and CSS variables |
| `src/utils/api.js` | API client functions for backend communication |
| `vite.config.js` | Vite bundler configuration |
| `package.json` | Node.js dependencies and scripts |

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | This comprehensive documentation |
| `QUICKSTART.md` | Quick start guide for beginners |
| `HUGGINGFACE_MODEL_GUIDE.md` | Detailed HF model integration guide |
| `IMAGE_PREDICTION_FIX.md` | Troubleshooting for image prediction issues |
| `TRAIN_MODELS.md` | Guide for training custom models |

### Test Files

| File | Purpose |
|------|---------|
| `test_huggingface_model.py` | Test HF model loading and inference |
| `test_image_prediction.py` | Test complete image prediction pipeline |

---

## ⚠️ Medical Disclaimer

**CRITICAL NOTICE**: This application is designed **exclusively for educational and demonstration purposes**. 

**DO NOT USE THIS APPLICATION FOR**:
- Actual medical diagnosis
- Treatment decisions
- Self-diagnosis
- Emergency medical situations

**ALWAYS**:
- Consult qualified healthcare professionals
- Seek proper medical diagnosis from radiologists
- Follow professional medical advice
- Get emergency medical care when needed

**AI Limitations**:
- Models may produce false positives/negatives
- Not validated for clinical use
- Not FDA approved
- Should not replace professional medical judgment

By using this application, you acknowledge these limitations and agree to use it only for educational purposes.

---

## 🙏 Credits

### Development Team
This project was developed as an educational demonstration of AI in healthcare.

**Developed at**: Dawood University of Engineering & Technology

### Technologies & Models
- **Hugging Face**: For the Vision Transformer model
- **PyTorch & Transformers**: Deep learning frameworks
- **React**: Frontend framework
- **Flask**: Backend framework
- **Medical Datasets**: Public chest X-ray datasets from Kaggle and NIH

### Acknowledgments
- Medical AI research community
- Open-source contributors
- Healthcare professionals who inspire AI innovation

---

## 📞 Support & Contact

### Getting Help
1. **Check this README** for comprehensive documentation
2. **Review troubleshooting section** for common issues
3. **Check documentation files** for specific topics
4. **Review code comments** for implementation details

### Reporting Issues
If you encounter bugs or have suggestions:
1. Check existing issues first
2. Provide detailed error messages
3. Include steps to reproduce
4. Specify your environment (OS, Python version, etc.)

---

## 📄 License

This project is provided **as-is** for educational purposes.

When using or modifying this project:
- Maintain the medical disclaimer
- Credit original sources
- Use responsibly
- Do not use for actual medical diagnosis

---

## 🚀 Future Enhancements

Potential improvements for learning purposes:
- **Multi-class classification**: Add COVID-19, TB, Lung Cancer detection
- **Model ensemble**: Combine multiple models for better accuracy
- **User authentication**: Add login/signup functionality
- **History tracking**: Save prediction history
- **Report generation**: PDF export of analysis results
- **Mobile app**: Native iOS/Android versions
- **Real-time chat**: Medical Q&A chatbot
- **Data visualization**: Analytics dashboard

---

## 🎯 Learning Outcomes

By studying this project, you'll learn:
- Full-stack web development (React + Flask)
- AI/ML integration in web applications
- Computer vision with Vision Transformers
- REST API design and implementation
- State management in React
- Python backend development
- Model deployment strategies
- UI/UX design principles
- Error handling and validation
- Testing and debugging techniques

---

**Built with ❤️ to demonstrate the potential of AI in healthcare**

*Last Updated: December 2025*
