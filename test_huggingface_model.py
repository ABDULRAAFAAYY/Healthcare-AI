"""
Test script for Hugging Face X-ray model integration
This script tests the new HuggingFaceXRayPredictor
"""

import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from ml.huggingface_xray_predictor import HuggingFaceXRayPredictor

def test_model_loading():
    """Test if the Hugging Face model loads correctly"""
    print("=" * 60)
    print("Testing Hugging Face X-ray Model Integration")
    print("=" * 60)
    
    print("\n1. Initializing HuggingFaceXRayPredictor...")
    predictor = HuggingFaceXRayPredictor()
    
    print("\n2. Checking if model is ready...")
    if predictor.is_ready():
        print("   [OK] Model is ready!")
        
        print("\n3. Getting model information...")
        info = predictor.get_model_info()
        print(f"   Model Name: {info['model_name']}")
        print(f"   Device: {info['device']}")
        print(f"   Number of Labels: {info['num_labels']}")
        print(f"   Labels: {info['labels']}")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] Hugging Face model is working correctly!")
        print("=" * 60)
        
        print("\n[Next Steps]")
        print("   1. Install dependencies: pip install -r backend/requirements.txt")
        print("   2. Start the backend: python backend/app.py")
        print("   3. Upload an X-ray image to test predictions")
        
        return True
    else:
        print("   [X] Model failed to load")
        print("\n[WARNING] Possible issues:")
        print("   1. Internet connection required for first-time download")
        print("   2. Missing dependencies (transformers, torch)")
        print("   3. Insufficient disk space for model cache")
        
        print("\n📋 Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Install dependencies: pip install transformers torch torchvision")
        print("   3. Try running again - model will be cached after first download")
        
        return False

if __name__ == "__main__":
    try:
        success = test_model_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[X] Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
