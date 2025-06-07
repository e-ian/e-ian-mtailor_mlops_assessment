import onnxruntime as ort
import numpy as np
from PIL import Image
import logging
from typing import Tuple, List, Optional, Union
import os
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles image preprocessing for the classification model.
    """
    
    def __init__(self):
        """Initialize preprocessor with ImageNet statistics."""
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.target_size = (224, 224)
        
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            
        Returns:
            np.ndarray: Preprocessed image tensor
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                image = Image.open(image)
            
            # Convert PIL Image to numpy array
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = image.resize(self.target_size, Image.BILINEAR)
                image = np.array(image, dtype=np.float32)

            if not isinstance(image, np.ndarray):
                raise TypeError(f"Unsupported image type: {type(image)}")
            
            # Ensure image is in correct format (H, W, C)
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = np.stack([image, image, image], axis=-1)
            elif len(image.shape) == 3 and image.shape[-1] == 4:
                # RGBA to RGB
                image = image[:, :, :3]
            
            # Ensure correct shape
            if image.shape[:2] != self.target_size:
                # Resize using nearest neighbor interpolation for numpy arrays
                from scipy.ndimage import zoom
                scale_h = self.target_size[0] / image.shape[0]
                scale_w = self.target_size[1] / image.shape[1]
                image = zoom(image, (scale_h, scale_w, 1), order=1)
            
            # Normalize pixel values to [0, 1]
            if image.max() > 1.0:
                image = image / 255.0
            
            # Normalize with ImageNet statistics
            image = (image - self.mean) / self.std
            
            # Convert to CHW format and add batch dimension
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            image = np.expand_dims(image, axis=0)   # Add batch dimension
            
            return image.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def preprocess_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            np.ndarray: Batch of preprocessed image tensors
        """
        try:
            processed_images = []
            for image in images:
                processed_image = self.preprocess_image(image)
                processed_images.append(processed_image[0])  # Remove batch dimension
            
            # Stack into batch
            return np.stack(processed_images, axis=0)
            
        except Exception as e:
            logger.error(f"Error preprocessing image batch: {str(e)}")
            raise


class ONNXClassifier:
    """
    ONNX model wrapper for image classification.
    """
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        Initialize ONNX classifier.
        
        Args:
            model_path (str): Path to ONNX model file
            providers (List[str], optional): ONNX Runtime execution providers
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.preprocessor = ImagePreprocessor()
        
        # Set default providers
        if providers is None:
            providers = ['CPUExecutionProvider']
            # Add GPU provider if available
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
        
        self.providers = providers
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and initialize session."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
            
            logger.info(f"Loading ONNX model from {self.model_path}")
            logger.info(f"Available providers: {ort.get_available_providers()}")
            logger.info(f"Using providers: {self.providers}")
            
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(
                self.model_path,
                providers=self.providers
            )
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Log model information
            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Input name: {self.input_name}, shape: {input_shape}")
            logger.info(f"Output name: {self.output_name}, shape: {output_shape}")
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            raise
    
    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[int, float, np.ndarray]:
        """
        Predict class for a single image.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            
        Returns:
            Tuple[int, float, np.ndarray]: (class_id, confidence, probabilities)
        """
        try:
            start_time = time.time()
            
            # Preprocess image
            processed_image = self.preprocessor.preprocess_image(image)
            
            # Run inference
            ort_inputs = {self.input_name: processed_image}
            outputs = self.session.run([self.output_name], ort_inputs)
            logits = outputs[0]
            
            # Apply softmax to get probabilities
            probabilities = self._softmax(logits[0])
            
            # Get prediction
            class_id = int(np.argmax(probabilities))
            confidence = float(probabilities[class_id])
            
            inference_time = time.time() - start_time
            logger.debug(f"Inference completed in {inference_time*1000:.2f} ms")
            
            return class_id, confidence, probabilities
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> List[Tuple[int, float, np.ndarray]]:
        """
        Predict classes for a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List[Tuple[int, float, np.ndarray]]: List of (class_id, confidence, probabilities)
        """
        try:
            start_time = time.time()
            
            # Preprocess batch
            processed_batch = self.preprocessor.preprocess_batch(images)
            
            # Run batch inference
            ort_inputs = {self.input_name: processed_batch}
            outputs = self.session.run([self.output_name], ort_inputs)
            logits_batch = outputs[0]
            
            # Process results
            results = []
            for logits in logits_batch:
                probabilities = self._softmax(logits)
                class_id = int(np.argmax(probabilities))
                confidence = float(probabilities[class_id])
                results.append((class_id, confidence, probabilities))
            
            inference_time = time.time() - start_time
            logger.debug(f"Batch inference ({len(images)} images) completed in {inference_time*1000:.2f} ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise
    
    def get_top_k_predictions(self, image: Union[str, Image.Image, np.ndarray], k: int = 5) -> List[Tuple[int, float]]:
        """
        Get top-k predictions for an image.
        
        Args:
            image: Input image
            k (int): Number of top predictions to return
            
        Returns:
            List[Tuple[int, float]]: List of (class_id, confidence) tuples
        """
        try:
            _, _, probabilities = self.predict(image)
            
            # Get top-k indices
            top_k_indices = np.argsort(probabilities)[-k:][::-1]
            
            # Create results
            results = []
            for idx in top_k_indices:
                results.append((int(idx), float(probabilities[idx])))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting top-k predictions: {str(e)}")
            raise
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """
        Apply softmax activation function.
        
        Args:
            x (np.ndarray): Input logits
            
        Returns:
            np.ndarray: Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    
    def warmup(self, num_iterations: int = 3):
        """
        Warm up the model with dummy inputs.
        
        Args:
            num_iterations (int): Number of warmup iterations
        """
        try:
            logger.info(f"Warming up model with {num_iterations} iterations...")
            
            # Create dummy input
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            
            start_time = time.time()
            for i in range(num_iterations):
                ort_inputs = {self.input_name: dummy_input}
                _ = self.session.run([self.output_name], ort_inputs)
            
            warmup_time = time.time() - start_time
            logger.info(f"Warmup completed in {warmup_time*1000:.2f} ms")
            
        except Exception as e:
            logger.error(f"Error during warmup: {str(e)}")
            raise
    
    def benchmark(self, num_iterations: int = 100) -> float:
        """
        Benchmark model inference speed.
        
        Args:
            num_iterations (int): Number of benchmark iterations
            
        Returns:
            float: Average inference time in milliseconds
        """
        try:
            logger.info(f"Benchmarking model with {num_iterations} iterations...")
            
            # Warm up first
            self.warmup()
            
            # Create dummy input
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                ort_inputs = {self.input_name: dummy_input}
                _ = self.session.run([self.output_name], ort_inputs)
            
            total_time = time.time() - start_time
            avg_time = (total_time / num_iterations) * 1000  # Convert to milliseconds
            
            logger.info(f"Average inference time: {avg_time:.2f} ms")
            return avg_time
            
        except Exception as e:
            logger.error(f"Error during benchmarking: {str(e)}")
            raise


def load_classifier(model_path: str, providers: Optional[List[str]] = None) -> ONNXClassifier:
    """
    Factory function to create and load ONNX classifier.
    
    Args:
        model_path (str): Path to ONNX model file
        providers (List[str], optional): ONNX Runtime execution providers
        
    Returns:
        ONNXClassifier: Loaded classifier instance
    """
    return ONNXClassifier(model_path, providers)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ONNX model inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--image_path', type=str,
                        help='Path to test image')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    
    args = parser.parse_args()
    
    try:
        # Load classifier
        classifier = load_classifier(args.model_path)
        
        if args.image_path:
            # Test prediction
            logger.info(f"Testing prediction on {args.image_path}")
            class_id, confidence, probabilities = classifier.predict(args.image_path)
            
            logger.info(f"Predicted class: {class_id}")
            logger.info(f"Confidence: {confidence:.4f}")
            
            # Get top-5 predictions
            top_5 = classifier.get_top_k_predictions(args.image_path, k=5)
            logger.info("Top-5 predictions:")
            for i, (cls_id, conf) in enumerate(top_5):
                logger.info(f"  {i+1}. Class {cls_id}: {conf:.4f}")
        
        if args.benchmark:
            # Run benchmark
            avg_time = classifier.benchmark()
            logger.info(f"Benchmark completed - Average inference time: {avg_time:.2f} ms")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        exit(1)
