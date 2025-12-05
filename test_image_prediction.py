"""
Quick test script to verify image prediction is working
"""

import sys
import os
from io import BytesIO
from PIL import Image

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from ml.image_predictor import ImagePredictor

def create_test_image():
    """Create a simple test X-ray-like image"""
    # Create a grayscale image (simulating an X-ray)
    img = Image.new('L', (224, 224), color=128)
    
    # Save to BytesIO to simulate file upload
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    img_bytes.name = 'test_xray.jpg'
    
    # Add stream attribute to mimic Flask file upload
    class FileWrapper:
        def __init__(self, bytes_io):
            self.stream = bytes_io
            self.filename = 'test_xray.jpg'
        
        def seek(self, pos):
            self.stream.seek(pos)
    
    return FileWrapper(img_bytes)

def test_image_prediction():
    """Test the image prediction pipeline"""
    print("=" * 60)
    print("Testing Image Prediction Pipeline")
    print("=" * 60)
    
    print("\n1. Initializing ImagePredictor...")
    predictor = ImagePredictor()
    
    print(f"\n2. Predictor ready: {predictor.is_ready()}")
    
    print("\n3. Creating test image...")
    test_image = create_test_image()
    
    print("\n4. Running prediction...")
    result = predictor.predict(test_image)
    
    print("\n5. Result:")
    print(f"   Success: {result.get('success', False)}")
    
    if result.get('success'):
        print(f"   Model Type: {result.get('model_type', 'unknown')}")
        print(f"   Number of Predictions: {len(result.get('predictions', []))}")
        
        if 'predictions' in result and len(result['predictions']) > 0:
            top_prediction = result['predictions'][0]
            print(f"\n   Top Prediction:")
            print(f"      Condition: {top_prediction.get('condition', 'N/A')}")
            print(f"      Confidence: {top_prediction.get('confidence', 0):.2%}")
            print(f"      Severity: {top_prediction.get('severity', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] Image prediction is working!")
        print("=" * 60)
        return True
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")
        print("\n" + "=" * 60)
        print("[FAILED] Image prediction failed")
        print("=" * 60)
        return False

if __name__ == "__main__":
    try:
        success = test_image_prediction()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
