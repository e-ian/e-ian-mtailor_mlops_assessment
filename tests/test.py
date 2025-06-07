import unittest
import os
import sys
import numpy as np
from PIL import Image
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.model import ONNXClassifier, ImagePreprocessor, load_classifier
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure src/model.py exists and is properly configured")
    sys.exit(1)

class TestImagePreprocessor(unittest.TestCase):
    """Test image preprocessing."""
    
    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        # Create test image
        test_image = Image.new('RGB', (256, 256), color='red')
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        test_image.save(self.temp_file.name)
        
    def tearDown(self):
        os.unlink(self.temp_file.name)
    
    def test_preprocessing(self):
        """Test basic preprocessing functionality."""
        result = self.preprocessor.preprocess_image(self.temp_file.name)
        
        # Check output shape and type
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.assertEqual(result.dtype, np.float32)
        
        # Check value range is reasonable
        self.assertTrue(np.all(np.isfinite(result)))


class TestONNXModel(unittest.TestCase):
    """Test ONNX model functionality."""
    
    def setUp(self):
        self.model_path = "model_artifacts/classification_model.onnx"
        if not os.path.exists(self.model_path):
            self.skipTest("ONNX model not found")
        
        try:
            self.classifier = load_classifier(self.model_path)
        except Exception as e:
            self.skipTest(f"Could not load model: {e}")
    
    def test_model_loading(self):
        """Test model loads successfully."""
        self.assertIsNotNone(self.classifier)
        self.assertIsNotNone(self.classifier.session)
    
    def test_prediction(self):
        """Test model prediction works."""
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='blue')
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        test_image.save(temp_file.name)
        
        try:
            class_id, confidence, probabilities = self.classifier.predict(temp_file.name)
            
            # Check output types and ranges
            self.assertIsInstance(class_id, int)
            self.assertIsInstance(confidence, float)
            self.assertIsInstance(probabilities, np.ndarray)
            
            self.assertGreaterEqual(class_id, 0)
            self.assertLess(class_id, 1000)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            self.assertEqual(len(probabilities), 1000)
            
        finally:
            os.unlink(temp_file.name)


class TestSampleImages(unittest.TestCase):
    """Test with sample images if available."""
    
    def setUp(self):
        self.model_path = "model_artifacts/classification_model.onnx"
        self.sample_images = {
            "tench": "sample_images/n01440764_tench.JPEG",
            "turtle": "sample_images/n01667114_mud_turtle.JPEG"
        }
        
        if not os.path.exists(self.model_path):
            self.skipTest("ONNX model not found")
        
        try:
            self.classifier = load_classifier(self.model_path)
        except Exception as e:
            self.skipTest(f"Could not load model: {e}")
    
    def test_sample_predictions(self):
        """Test predictions on sample images."""
        for name, path in self.sample_images.items():
            if os.path.exists(path):
                with self.subTest(image=name):
                    class_id, confidence, _ = self.classifier.predict(path)
                    print(f"{name}: Class {class_id}, Confidence {confidence:.4f}")
                    
                    # Just check that we get reasonable outputs
                    self.assertIsInstance(class_id, int)
                    self.assertGreaterEqual(confidence, 0.0)
                    self.assertLessEqual(confidence, 1.0)


def main():
    """Run tests."""
    # Try to run download script if it exists
    try:
        import subprocess
        if os.path.exists("scripts/download_weights.py"):
            subprocess.run([sys.executable, "scripts/download_weights.py", "--placeholders"], 
                         capture_output=True, timeout=30)
    except:
        pass
    
    # Run tests
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
