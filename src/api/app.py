"""
FastAPI Application for Diabetic Retinopathy Detection

This module provides a REST API for DR grading inference:
- Single image prediction
- Batch prediction
- Health check endpoint
- Grad-CAM visualization endpoint
"""

import io
import sys
from pathlib import Path
from typing import Optional, List
import base64
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import numpy as np
import cv2

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Lazy imports for model
predictor = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="AI-powered grading of Diabetic Retinopathy from fundus images",
    version="1.0.0",
)


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    class_id: int
    class_name: str
    confidence: float
    raw_value: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str


class GradCAMResponse(BaseModel):
    """Response model for Grad-CAM visualization."""
    prediction: PredictionResponse
    gradcam_image: str  # Base64 encoded image


def get_predictor():
    """Lazy load predictor."""
    global predictor
    
    if predictor is None:
        from inference import DRPredictor
        
        # Look for model checkpoint
        checkpoint_paths = [
            Path("checkpoints/best_model.ckpt"),
            Path("checkpoints/dr_detection/best.ckpt"),
            Path("model.onnx"),
        ]
        
        model_path = None
        for path in checkpoint_paths:
            if path.exists():
                model_path = str(path)
                break
        
        if model_path is None:
            raise RuntimeError(
                "No model found! Please place a checkpoint in 'checkpoints/' "
                "or an ONNX model as 'model.onnx'"
            )
        
        is_onnx = model_path.endswith(".onnx")
        
        predictor = DRPredictor(
            checkpoint_path=None if is_onnx else model_path,
            onnx_path=model_path if is_onnx else None,
            device="auto",
            use_tta=False,
        )
        
        logger.info(f"Model loaded from {model_path}")
    
    return predictor


@app.on_event("startup")
async def startup_event():
    """Load model on startup (optional, for faster first request)."""
    try:
        get_predictor()
        logger.info("Model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Model not pre-loaded: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        pred = get_predictor()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=str(pred.device),
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="unknown",
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict DR grade for a single fundus image.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        Prediction with class, name, and confidence
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG)"
        )
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image"
            )
        
        # Get predictor and make prediction
        pred = get_predictor()
        result = pred.predict(image, return_raw=True)
        
        return PredictionResponse(
            class_id=result["class"],
            class_name=result["class_name"],
            confidence=result["confidence"],
            raw_value=result.get("raw_value"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict DR grades for multiple images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of predictions
    """
    if len(files) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 images per batch"
        )
    
    pred = get_predictor()
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                result = pred.predict(image, return_raw=True)
                results.append({
                    "filename": file.filename,
                    "class_id": result["class"],
                    "class_name": result["class_name"],
                    "confidence": result["confidence"],
                    "raw_value": result.get("raw_value"),
                })
            else:
                results.append({
                    "filename": file.filename,
                    "error": "Could not decode image",
                })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
            })
    
    return JSONResponse(content={"predictions": results})


@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    Predict DR grade with Grad-CAM visualization.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction and base64-encoded Grad-CAM overlay
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG)"
        )
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image"
            )
        
        pred = get_predictor()
        
        # Make prediction
        result = pred.predict(image, return_raw=True)
        
        # Generate Grad-CAM (only if using PyTorch model)
        if not pred.use_onnx:
            from src.xai.gradcam import GradCAM
            import torch
            
            # Preprocess for Grad-CAM
            processed = pred._preprocess_image(image)
            tensor = pred._to_tensor(processed)
            
            # Generate heatmap
            gradcam = GradCAM(pred.model)
            heatmap = gradcam(tensor)
            
            # Create overlay
            overlay = gradcam.visualize(processed, heatmap)
            
            # Encode to base64
            _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            gradcam_b64 = base64.b64encode(buffer).decode('utf-8')
        else:
            gradcam_b64 = None
        
        return {
            "prediction": {
                "class_id": result["class"],
                "class_name": result["class_name"],
                "confidence": result["confidence"],
                "raw_value": result.get("raw_value"),
            },
            "gradcam_image": gradcam_b64,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/classes")
async def get_classes():
    """Get class information."""
    return {
        "classes": [
            {"id": 0, "name": "No DR", "description": "No apparent retinopathy"},
            {"id": 1, "name": "Mild NPDR", "description": "Microaneurysms only"},
            {"id": 2, "name": "Moderate NPDR", "description": "More than microaneurysms"},
            {"id": 3, "name": "Severe NPDR", "description": "4-2-1 rule applies"},
            {"id": 4, "name": "Proliferative DR", "description": "Neovascularization"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
