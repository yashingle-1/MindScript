"""
MindScript API
==============
FastAPI server for cognitive pattern analysis.

Author: [Your Name]
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import torch
import time
import hashlib
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mindscript.models.mindscript_model import MindScriptModel

# Initialize FastAPI
app = FastAPI(
    title="MindScript API",
    description="Cognitive Pattern Analysis from Text",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = None


class TextInput(BaseModel):
    """Input schema for text analysis"""
    text: str = Field(..., min_length=10, description="Text to analyze")
    include_confidence: bool = Field(default=True, description="Include confidence score")


class DimensionScores(BaseModel):
    """Schema for dimension scores"""
    analytical: float = Field(..., ge=0, le=1)
    creative: float = Field(..., ge=0, le=1)
    social: float = Field(..., ge=0, le=1)
    structured: float = Field(..., ge=0, le=1)
    emotional: float = Field(..., ge=0, le=1)


class AnalysisResult(BaseModel):
    """Output schema for analysis"""
    dimensions: DimensionScores
    confidence: float
    dominant_dimension: str
    cognitive_archetype: str
    processing_time_ms: float
    timestamp: str
    text_length: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path("models/mindscript_best.pt")
    
    if model_path.exists():
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        model = MindScriptModel(
            encoder_name="distilbert-base-uncased",
            num_dimensions=5,
            hidden_dim=768
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ No trained model found. Please train the model first.")
        # Initialize without weights for demo
        model = MindScriptModel()
        model.to(device)
        model.eval()


@app.get("/", response_model=Dict)
async def root():
    """API root endpoint"""
    return {
        "name": "MindScript API",
        "version": "1.0.0",
        "description": "Cognitive Pattern Analysis from Text",
        "endpoints": {
            "/analyze": "POST - Analyze text for cognitive patterns",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        device=str(device),
        timestamp=datetime.now().isoformat()
    )


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_text(input_data: TextInput):
    """
    Analyze text for cognitive patterns.
    
    Returns dimension scores and cognitive archetype.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    start_time = time.time()
    
    try:
        # Run prediction
        result = model.predict(input_data.text, device=device)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Determine cognitive archetype
        archetype = _determine_archetype(result["dimensions"])
        
        return AnalysisResult(
            dimensions=DimensionScores(**result["dimensions"]),
            confidence=result["confidence"],
            dominant_dimension=result["dominant_dimension"],
            cognitive_archetype=archetype,
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat(),
            text_length=len(input_data.text)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_analyze")
async def batch_analyze(texts: List[str]):
    """Analyze multiple texts"""
    results = []
    
    for text in texts:
        try:
            result = await analyze_text(TextInput(text=text))
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "text": text[:50]})
    
    return {"results": results, "count": len(results)}


def _determine_archetype(dimensions: Dict[str, float]) -> str:
    """Determine cognitive archetype from dimension scores"""
    
    # Find top 2 dimensions
    sorted_dims = sorted(dimensions.items(), key=lambda x: x[1], reverse=True)
    top_two = [d[0] for d in sorted_dims[:2]]
    
    # Archetype mappings
    archetypes = {
        ("analytical", "structured"): "The Strategist",
        ("analytical", "creative"): "The Innovator",
        ("creative", "social"): "The Visionary",
        ("creative", "emotional"): "The Artist",
        ("social", "emotional"): "The Empath",
        ("social", "creative"): "The Collaborator",
        ("structured", "analytical"): "The Architect",
        ("structured", "social"): "The Organizer",
        ("emotional", "social"): "The Connector",
        ("emotional", "creative"): "The Dreamer"
    }
    
    key = tuple(sorted(top_two))
    return archetypes.get(key, "The Explorer")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)