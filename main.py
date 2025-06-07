from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import logging
import os
import io
import base64
import sys
from typing import Optional, Dict, Any
import traceback
from PIL import Image
import numpy as np

sys.path.append('/app')
sys.path.append('/app/src')
sys.path.append('./src')
sys.path.append('.')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Classification API", version="1.0.0")

classifier = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup with robust import handling."""
    global classifier
    
    try:
        logger.info("Starting up the application...")
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Contents of /app: {os.listdir('/app') if os.path.exists('/app') else 'N/A'}")
        logger.info(f"Contents of /app/src: {os.listdir('/app/src') if os.path.exists('/app/src') else 'N/A'}")
        
        # Strategy 1: Try direct import from src
        try:
            from src.model import ONNXClassifier, load_classifier
            logger.info("Successfully imported from src.model")
        except ImportError as e1:
            logger.warning(f"Import from src.model failed: {e1}")
            
            # Strategy 2: Try importing without src prefix
            try:
                from model import ONNXClassifier, load_classifier
                logger.info("Successfully imported from model")
            except ImportError as e2:
                logger.warning(f"Import from model failed: {e2}")
                
                # Strategy 3: Try dynamic import with full path resolution
                try:
                    import importlib.util
                    
                    # Find model.py file
                    model_file_paths = [
                        '/app/src/model.py',
                        '/app/model.py',
                        './src/model.py',
                        './model.py',
                        'src/model.py',
                        'model.py'
                    ]
                    
                    model_file = None
                    for path in model_file_paths:
                        if os.path.exists(path):
                            model_file = path
                            logger.info(f"Found model file at: {path}")
                            break
                    
                    if not model_file:
                        # List all Python files to help debug
                        logger.error("Could not find model.py. Searching for Python files...")
                        for root, dirs, files in os.walk('/app'):
                            for file in files:
                                if file.endswith('.py'):
                                    logger.info(f"Found Python file: {os.path.join(root, file)}")
                        raise ImportError("model.py not found in any expected location")
                    
                    # Load module dynamically
                    spec = importlib.util.spec_from_file_location("model_module", model_file)
                    model_module = importlib.util.module_from_spec(spec)
                    
                    # Add to sys.modules to make it importable
                    sys.modules["model_module"] = model_module
                    spec.loader.exec_module(model_module)
                    
                    # Extract classes
                    ONNXClassifier = model_module.ONNXClassifier
                    load_classifier = model_module.load_classifier
                    logger.info(f"Successfully imported using dynamic import from {model_file}")
                    
                except Exception as e3:
                    logger.error(f"Dynamic import failed: {e3}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise ImportError(f"All import strategies failed. Last error: {e3}")
        
        # Find the ONNX model file
        model_file_paths = [
            '/app/model_artifacts/classification_model.onnx',
            '/app/src/classification_model.onnx',
            '/app/classification_model.onnx',
            './model_artifacts/classification_model.onnx',
            './src/classification_model.onnx',
            './classification_model.onnx',
            'model_artifacts/classification_model.onnx',
            'src/classification_model.onnx',
            'classification_model.onnx'
        ]
        
        model_file = None
        for path in model_file_paths:
            if os.path.exists(path):
                model_file = path
                logger.info(f"Found ONNX model at: {path}")
                break
        
        if not model_file:
            # Search for any .onnx files
            logger.error("ONNX model not found. Searching for .onnx files...")
            for root, dirs, files in os.walk('/app'):
                for file in files:
                    if file.endswith('.onnx'):
                        full_path = os.path.join(root, file)
                        logger.info(f"Found ONNX file: {full_path}")
                        if not model_file:
                            model_file = full_path
            
            if not model_file:
                raise FileNotFoundError("No ONNX model file found")
        
        logger.info(f"Loading model from: {model_file}")
        
        # Load the classifier
        classifier = load_classifier(model_file)
        logger.info("Model loaded successfully!")
        
        # Test the model with a dummy prediction
        test_image = Image.new('RGB', (224, 224), color='red')
        class_id, confidence, _ = classifier.predict(test_image)
        logger.info(f"Model test successful - Class: {class_id}, Confidence: {confidence:.4f}")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    global classifier
    
    status = {
        "status": "healthy" if classifier is not None else "unhealthy",
        "model_loaded": classifier is not None,
        "message": "Service is running",
        "debug_info": {
            "working_directory": os.getcwd(),
            "python_path": sys.path[:3],
            "app_contents": os.listdir('/app') if os.path.exists('/app') else [],
            "src_contents": os.listdir('/app/src') if os.path.exists('/app/src') else []
        }
    }
    
    if classifier is not None:
        try:
            model_info = classifier.get_model_info()
            status["model_info"] = {
                "input_shape": model_info.get("input_shape"),
                "output_shape": model_info.get("output_shape"),
                "has_preprocessing": model_info.get("has_preprocessing"),
                "model_path": model_info.get("model_path")
            }
        except Exception as e:
            status["model_info_error"] = str(e)
    
    return status


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict image class from uploaded file.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction results
    """
    global classifier
    
    if classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check /health endpoint for details."
        )
    
    try:
        image_data = await file.read()
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")

        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        class_id, confidence, probabilities = classifier.predict(image)

        top5_indices = np.argsort(probabilities)[-5:][::-1]
        top5_predictions = [
            {
                "class_id": int(idx),
                "confidence": float(probabilities[idx])
            }
            for idx in top5_indices
        ]
        
        return {
            "success": True,
            "predicted_class": int(class_id),
            "confidence": float(confidence),
            "top5_predictions": top5_predictions,
            "filename": file.filename,
            "file_size": len(image_data),
            "image_size": image.size
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_base64")
async def predict_base64(request: Dict[str, Any]):
    """
    Predict image class from base64 encoded image.
    
    Args:
        request: JSON with base64 encoded image
        
    Returns:
        JSON response with prediction results
    """
    global classifier
    
    if classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check /health endpoint for details."
        )
    
    try:
        image_b64 = request.get("image")
        if not image_b64:
            raise HTTPException(status_code=400, detail="No image data provided")
        try:
            image_data = base64.b64decode(image_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        class_id, confidence, probabilities = classifier.predict(image)
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        top5_predictions = [
            {
                "class_id": int(idx),
                "confidence": float(probabilities[idx])
            }
            for idx in top5_indices
        ]
        
        return {
            "success": True,
            "predicted_class": int(class_id),
            "confidence": float(confidence),
            "top5_predictions": top5_predictions,
            "image_size": image.size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model_info")
def get_model_info():
    """Get model information."""
    global classifier
    
    if classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check /health endpoint for details."
        )
    
    try:
        info = classifier.get_model_info()
        return {
            "success": True,
            "model_info": info
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.post("/hello")
def hello():
    """Hello endpoint for testing."""
    return {
        "message": "Hello Cerebrium! Classification service is running.",
        "model_loaded": classifier is not None,
        "service_status": "operational"
    }


@app.get("/")
def root():
    """Root endpoint with API information."""
    global classifier
    
    return {
        "message": "Image Classification API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": classifier is not None,
        "endpoints": {
            "health": "GET /health - Health check and debug info",
            "predict": "POST /predict - Upload file for prediction",
            "predict_base64": "POST /predict_base64 - Base64 image prediction",
            "model_info": "GET /model_info - Model metadata",
            "hello": "POST /hello - Basic connectivity test"
        },
        "usage": {
            "file_upload": "curl -X POST /predict -F 'file=@image.jpg'",
            "base64": "curl -X POST /predict_base64 -H 'Content-Type: application/json' -d '{\"image\":\"base64_data\"}'"
        }
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Request: {request.method} {request.url}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8192)
