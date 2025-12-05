import os
import numpy as np
from PIL import Image
import io
from .model_loader import ModelLoader
from .huggingface_xray_predictor import HuggingFaceXRayPredictor

class ImagePredictor:
    """Handles medical image-based prediction (X-ray analysis)"""
    
    def __init__(self):
        """Initialize the image predictor and load model"""
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_path = os.path.join(self.base_path, 'models')
        
        # Try to load Hugging Face model first
        print("Attempting to load Hugging Face X-ray model...")
        self.hf_predictor = HuggingFaceXRayPredictor()
        
        # If HF model fails, try Keras model
        if not self.hf_predictor.is_ready():
            print("Hugging Face model not available, trying Keras model...")
            self.model = ModelLoader.load_keras_model(
                os.path.join(self.models_path, 'xray_model.h5')
            )
        else:
            print("[OK] Hugging Face model loaded successfully!")
            self.model = None  # We'll use HF model instead
        
        # Image preprocessing parameters
        self.img_size = (224, 224)  # Standard size for medical image models
        
        # Demo classification labels
        self.demo_labels = [
            'Normal',
            'Pneumonia',
            'COVID-19',
            'Tuberculosis',
            'Lung Cancer'
        ]
        
        # Detailed information for each condition
        self.condition_info = {
            'Normal': {
                'description': 'No abnormalities detected in the X-ray',
                'severity': 'None',
                'recommendations': [
                    'Continue regular health check-ups',
                    'Maintain a healthy lifestyle',
                    'No immediate action required'
                ]
            },
            'Pneumonia': {
                'description': 'Inflammation of the lungs, typically caused by infection',
                'severity': 'Moderate to Severe',
                'recommendations': [
                    'Consult a pulmonologist immediately',
                    'Antibiotics may be required',
                    'Rest and hydration are essential',
                    'Monitor oxygen levels'
                ]
            },
            'COVID-19': {
                'description': 'Respiratory illness caused by SARS-CoV-2 virus',
                'severity': 'Moderate to Severe',
                'recommendations': [
                    'Isolate immediately',
                    'Get PCR test confirmation',
                    'Monitor oxygen saturation',
                    'Seek immediate medical care if breathing worsens',
                    'Follow local health guidelines'
                ]
            },
            'Tuberculosis': {
                'description': 'Bacterial infection primarily affecting the lungs',
                'severity': 'Severe',
                'recommendations': [
                    'Immediate medical consultation required',
                    'Start TB treatment regimen as prescribed',
                    'Isolate to prevent transmission',
                    'Complete full course of antibiotics (6-9 months)',
                    'Regular follow-up X-rays needed'
                ]
            },
            'Lung Cancer': {
                'description': 'Abnormal cell growth in lung tissue',
                'severity': 'Severe',
                'recommendations': [
                    'Urgent consultation with oncologist',
                    'Further diagnostic tests required (CT, biopsy)',
                    'Discuss treatment options (surgery, chemo, radiation)',
                    'Seek second opinion',
                    'Consider support groups and counseling'
                ]
            }
        }
    
    def is_ready(self):
        """Check if the predictor is ready"""
        return True
    
    def preprocess_image(self, image_file):
        """Preprocess the uploaded image for model prediction"""
        # Read image
        img = Image.open(image_file.stream)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(self.img_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_file):
        """Predict condition from medical image"""
        try:
            # Reset file stream to beginning (important for file uploads)
            if hasattr(image_file, 'seek'):
                image_file.seek(0)
            
            # Try Hugging Face model first
            if hasattr(self, 'hf_predictor') and self.hf_predictor.is_ready():
                print("Using Hugging Face model for prediction...")
                # Reset stream again before HF predictor reads it
                if hasattr(image_file, 'seek'):
                    image_file.seek(0)
                return self.hf_predictor.predict(image_file)
            
            # Fall back to Keras model
            if self.model is not None:
                print("Using Keras model for prediction...")
                # Reset stream for Keras model
                if hasattr(image_file, 'seek'):
                    image_file.seek(0)
                # Preprocess image
                img_array = self.preprocess_image(image_file)
                return self._predict_with_model(img_array)
            
            # Last resort: demo prediction
            print("Using demo mode for prediction...")
            # Reset stream for demo mode
            if hasattr(image_file, 'seek'):
                image_file.seek(0)
            img_array = self.preprocess_image(image_file)
            return self._demo_predict(img_array)
                
        except Exception as e:
            print(f"Image prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Failed to process image: {str(e)}'
            }
    
    def _predict_with_model(self, img_array):
        """Make prediction using trained model"""
        # Get prediction probabilities
        predictions = self.model.predict(img_array)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for idx in top_indices:
            condition = self.demo_labels[idx] if idx < len(self.demo_labels) else f'Condition_{idx}'
            confidence = float(predictions[idx])
            
            condition_data = self.condition_info.get(condition, {
                'description': 'Medical condition detected',
                'severity': 'Unknown',
                'recommendations': ['Consult a healthcare professional']
            })
            
            results.append({
                'condition': condition,
                'confidence': confidence,
                'description': condition_data['description'],
                'severity': condition_data['severity'],
                'recommendations': condition_data['recommendations']
            })
        
        return {
            'success': True,
            'predictions': results,
            'model_type': 'trained',
            'image_size': self.img_size
        }
    
    def _demo_predict(self, img_array=None):
        """Fallback demo prediction when model isn't available"""
        # Generate more realistic demo predictions based on simple image analysis
        # This is still a demo and should not be used for actual diagnosis
        
        results = []
        
        # Create weighted probabilities that favor "Normal" in demo mode
        # This is more realistic than random predictions
        if img_array is not None:
            # Simple brightness analysis as a basic heuristic
            avg_brightness = np.mean(img_array)
            
            # Adjust probabilities based on brightness (very basic heuristic)
            if avg_brightness > 0.6:
                # Brighter images more likely to be normal
                base_probs = [0.65, 0.15, 0.10, 0.05, 0.05]  # Favor Normal
            elif avg_brightness < 0.3:
                # Darker images might indicate abnormalities
                base_probs = [0.30, 0.25, 0.20, 0.15, 0.10]
            else:
                # Medium brightness
                base_probs = [0.45, 0.20, 0.15, 0.12, 0.08]
        else:
            # Default conservative probabilities
            base_probs = [0.60, 0.15, 0.12, 0.08, 0.05]
        
        # Add small random variation to make it look realistic
        noise = np.random.uniform(-0.05, 0.05, len(base_probs))
        demo_probabilities = np.array(base_probs) + noise
        demo_probabilities = np.maximum(demo_probabilities, 0)  # Ensure non-negative
        demo_probabilities = demo_probabilities / demo_probabilities.sum()  # Normalize
        
        # Sort by probability
        sorted_indices = np.argsort(demo_probabilities)[::-1]
        
        for idx in sorted_indices[:3]:
            condition = self.demo_labels[idx]
            confidence = float(demo_probabilities[idx])
            
            condition_data = self.condition_info[condition]
            
            results.append({
                'condition': condition,
                'confidence': confidence,
                'description': condition_data['description'],
                'severity': condition_data['severity'],
                'recommendations': condition_data['recommendations']
            })
        
        return {
            'success': True,
            'predictions': results,
            'model_type': 'demo',
            'note': '⚠️ DEMO MODE: Using simulated predictions. Train the X-ray model with real medical images for accurate results.',
            'warning': '🚨 IMPORTANT: This is NOT a real medical diagnosis. These are simulated predictions for demonstration only. Always consult a qualified radiologist for actual X-ray interpretation.'
        }
