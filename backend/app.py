from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.predictor import SymptomPredictor
from ml.image_predictor import ImagePredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize predictors
symptom_predictor = SymptomPredictor()
image_predictor = ImagePredictor()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Healthcare AI API is running',
        'version': '1.0.0'
    })

@app.route('/api/predict/symptoms', methods=['POST'])
def predict_disease():
    """
    Predict disease based on symptoms
    Expected JSON: {"symptoms": ["fever", "cough", "fatigue"]}
    """
    try:
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({
                'error': 'Missing symptoms in request body'
            }), 400
        
        symptoms = data['symptoms']
        
        if not isinstance(symptoms, list) or len(symptoms) == 0:
            return jsonify({
                'error': 'Symptoms must be a non-empty list'
            }), 400
        
        # Get prediction
        result = symptom_predictor.predict(symptoms)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/predict/image', methods=['POST'])
def predict_image():
    """
    Predict disease from medical image (X-ray)
    Expected: multipart/form-data with 'image' file
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided'
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                'error': 'Empty filename'
            }), 400
        
        # Get prediction
        result = image_predictor.predict(image_file)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Image prediction failed: {str(e)}'
        }), 500

@app.route('/api/symptoms/list', methods=['GET'])
def get_symptoms_list():
    """Get list of available symptoms"""
    try:
        symptoms = symptom_predictor.get_available_symptoms()
        return jsonify({
            'symptoms': symptoms,
            'count': len(symptoms)
        }), 200
    except Exception as e:
        return jsonify({
            'error': f'Failed to retrieve symptoms: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Healthcare AI API Server")
    print("=" * 60)
    print(f"Symptom Predictor Status: {'Ready' if symptom_predictor.is_ready() else 'Not Ready'}")
    print(f"Image Predictor Status: {'Ready' if image_predictor.is_ready() else 'Not Ready'}")
    print("=" * 60)
    print("Server starting on http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
