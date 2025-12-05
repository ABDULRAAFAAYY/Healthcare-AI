"""
Hugging Face X-ray Classification Module
Uses pre-trained Vision Transformer (ViT) model for chest X-ray analysis
"""

import os
# Disable TensorFlow in transformers to avoid conflicts
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

import numpy as np
from PIL import Image
import torch

# Monkey-patch to disable TensorFlow checking in transformers
import sys
class FakeTF:
    """Fake TensorFlow module to prevent import errors"""
    Tensor = None
    
# Replace tensorflow in sys.modules to prevent transformers from using it
if 'tensorflow' in sys.modules:
    # Store original TF
    _original_tf = sys.modules['tensorflow']
    # Replace with fake
    sys.modules['tensorflow'] = FakeTF()

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import warnings
warnings.filterwarnings('ignore')

# Restore original TF if it existed
if 'tensorflow' in sys.modules and isinstance(sys.modules['tensorflow'], FakeTF):
    if '_original_tf' in locals():
        sys.modules['tensorflow'] = _original_tf
    else:
        del sys.modules['tensorflow']


class HuggingFaceXRayPredictor:
    """
    Handles X-ray image classification using Hugging Face pre-trained models
    Model: nickmuchi/vit-finetuned-chest-xray-pneumonia
    """
    
    def __init__(self, model_name="nickmuchi/vit-finetuned-chest-xray-pneumonia"):
        """
        Initialize the Hugging Face X-ray predictor
        
        Args:
            model_name: Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.model = None
        self.feature_extractor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Condition information mapping
        self.condition_info = {
            'NORMAL': {
                'display_name': 'Normal',
                'description': 'No abnormalities detected in the chest X-ray',
                'severity': 'None',
                'recommendations': [
                    'Continue regular health check-ups',
                    'Maintain a healthy lifestyle',
                    'No immediate action required'
                ]
            },
            'PNEUMONIA': {
                'display_name': 'Pneumonia',
                'description': 'Inflammation of the lungs, typically caused by infection',
                'severity': 'Moderate to Severe',
                'recommendations': [
                    'Consult a pulmonologist immediately',
                    'Antibiotics may be required',
                    'Rest and hydration are essential',
                    'Monitor oxygen levels',
                    'Get proper medical diagnosis and treatment'
                ]
            }
        }
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model and feature extractor"""
        try:
            print(f"Loading Hugging Face model: {self.model_name}")
            
            # Load feature extractor and model
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[OK] Model loaded successfully on {self.device}")
            print(f"[OK] Model labels: {self.model.config.id2label}")
            
        except Exception as e:
            print(f"[WARNING] Could not load Hugging Face model: {str(e)}")
            print("The system will fall back to demo mode.")
            self.model = None
            self.feature_extractor = None
    
    def is_ready(self):
        """Check if the model is loaded and ready"""
        return self.model is not None and self.feature_extractor is not None
    
    def preprocess_image(self, image):
        """
        Preprocess image for the model
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use the feature extractor to preprocess
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def predict(self, image_file):
        """
        Predict condition from chest X-ray image
        
        Args:
            image_file: File object containing the image
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Check if model is ready
            if not self.is_ready():
                return {
                    'success': False,
                    'error': 'Model not loaded. Please check your internet connection and try again.'
                }
            
            # Load and preprocess image
            image = Image.open(image_file.stream)
            inputs = self.preprocess_image(image)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                probabilities = probabilities.cpu().numpy()[0]
            
            # Get predictions
            results = []
            
            # Sort by probability (descending)
            sorted_indices = np.argsort(probabilities)[::-1]
            
            for idx in sorted_indices:
                label = self.model.config.id2label[idx]
                confidence = float(probabilities[idx])
                
                # Get condition info
                condition_data = self.condition_info.get(
                    label.upper(),
                    {
                        'display_name': label,
                        'description': 'Medical condition detected',
                        'severity': 'Unknown',
                        'recommendations': ['Consult a healthcare professional for proper diagnosis']
                    }
                )
                
                results.append({
                    'condition': condition_data['display_name'],
                    'confidence': confidence,
                    'description': condition_data['description'],
                    'severity': condition_data['severity'],
                    'recommendations': condition_data['recommendations']
                })
            
            return {
                'success': True,
                'predictions': results,
                'model_type': 'huggingface',
                'model_name': self.model_name,
                'device': str(self.device),
                'note': '[OK] Using pre-trained Hugging Face model for X-ray classification',
                'disclaimer': '[WARNING] MEDICAL DISCLAIMER: This AI model is for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis. Always consult a qualified radiologist or healthcare provider for accurate medical interpretation of X-ray images.'
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': f'Failed to process image: {str(e)}'
            }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_ready():
            return {
                'loaded': False,
                'message': 'Model not loaded'
            }
        
        return {
            'loaded': True,
            'model_name': self.model_name,
            'device': str(self.device),
            'labels': self.model.config.id2label,
            'num_labels': len(self.model.config.id2label)
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Hugging Face X-ray Predictor...")
    
    predictor = HuggingFaceXRayPredictor()
    
    if predictor.is_ready():
        print("\n✓ Model is ready!")
        print(f"Model info: {predictor.get_model_info()}")
    else:
        print("\n✗ Model failed to load")
