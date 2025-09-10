"""
FastAPI REST API for EduSpend TCA Prediction
Provides RESTful endpoints for model serving
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from typing import List, Optional
import uvicorn
import pickle
import os

# Initialize FastAPI app
app = FastAPI(
    title="EduSpend TCA Prediction API",
    description="REST API for predicting Total Cost of Attendance (TCA) for international education",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for TCA prediction"""
    country: str = Field(..., description="Country of education")
    program: str = Field(..., description="Study program")
    level: str = Field(..., description="Education level (Bachelors, Masters, PhD)")
    city: str = Field(..., description="City of education")
    duration_years: float = Field(..., ge=0.5, le=8.0, description="Duration in years")
    living_cost_index: int = Field(..., ge=30, le=200, description="Living cost index")
    rent_usd: int = Field(..., ge=200, le=5000, description="Monthly rent in USD")
    visa_fee_usd: int = Field(..., ge=0, le=2000, description="Visa fee in USD")
    insurance_usd: int = Field(..., ge=0, le=5000, description="Annual health insurance in USD")

class PredictionResponse(BaseModel):
    """Response model for TCA prediction"""
    predicted_tca: float = Field(..., description="Predicted Total Cost of Attendance in USD")
    confidence_score: Optional[float] = Field(None, description="Prediction confidence score")
    cost_breakdown: dict = Field(..., description="Breakdown of cost components")
    model_version: Optional[str] = Field(None, description="Model version used")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: str
    version: str

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    features: List[str]
    top_cities: List[str]
    model_performance: dict

# Global variables for model artifacts
model = None
top_cities = []
feature_names = []
model_name = "EduSpend-TCA-Predictor"

@app.on_event("startup")
async def load_model():
    """Load model on application startup"""
    global model, top_cities, feature_names
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Try to load from Model Registry first
        try:
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Model loaded from registry: {model_uri}")
        except:
            # Fallback to local artifacts
            print("⚠️  Model registry not available, using local artifacts")
            model = None
        
        # Load top cities list
        if os.path.exists("top_cities_list.pkl"):
            with open("top_cities_list.pkl", "rb") as f:
                top_cities = pickle.load(f)
        else:
            # Default top cities
            top_cities = ['New York', 'London', 'Toronto', 'Sydney', 'Boston', 
                         'San Francisco', 'Los Angeles', 'Chicago', 'Vancouver', 'Melbourne']
        
        # Load feature names
        if os.path.exists("feature_names.pkl"):
            with open("feature_names.pkl", "rb") as f:
                feature_names = pickle.load(f)
        else:
            feature_names = ['Country', 'Program', 'Level', 'City_Simplified', 
                           'Duration_Years', 'Living_Cost_Index', 'Rent_USD', 
                           'Visa_Fee_USD', 'Insurance_USD']
        
        print(f"✅ Model artifacts loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

def calculate_tca_fallback(request: PredictionRequest) -> float:
    """
    Fallback TCA calculation when ML model is not available
    Uses simplified business logic based on cost components
    """
    # Base calculation
    annual_rent = request.rent_usd * 12
    total_housing = annual_rent * request.duration_years
    
    # Country-based tuition estimates (simplified)
    country_tuition_base = {
        'USA': 35000, 'UK': 30000, 'Canada': 25000, 'Australia': 28000,
        'Germany': 5000, 'France': 8000, 'Netherlands': 12000, 'Sweden': 3000,
        'India': 3000, 'China': 5000, 'Japan': 15000, 'South Korea': 8000
    }
    
    # Program multipliers
    program_multipliers = {
        'Medicine': 1.8, 'Business': 1.3, 'Engineering': 1.1, 'Law': 1.4,
        'Computer Science': 1.1, 'Arts': 0.8, 'Science': 0.9, 'Social Sciences': 0.7
    }
    
    # Level multipliers
    level_multipliers = {
        'PhD': 1.2, 'Masters': 1.0, 'Bachelors': 0.8, 'Diploma': 0.6
    }
    
    # Calculate estimated tuition
    base_tuition = country_tuition_base.get(request.country, 20000)
    program_mult = program_multipliers.get(request.program, 1.0)
    level_mult = level_multipliers.get(request.level, 1.0)
    living_cost_adj = request.living_cost_index / 100
    
    estimated_tuition = base_tuition * program_mult * level_mult * living_cost_adj * request.duration_years
    
    # Total TCA
    total_tca = total_housing + estimated_tuition + request.insurance_usd + request.visa_fee_usd
    
    return max(total_tca, 5000)  # Minimum reasonable TCA

def prepare_model_input(request: PredictionRequest) -> pd.DataFrame:
    """Prepare input data for model prediction"""
    # Simplify city name
    city_simplified = request.city if request.city in top_cities else 'Other'
    
    # Create DataFrame
    input_data = pd.DataFrame([{
        'Country': request.country,
        'Program': request.program,
        'Level': request.level,
        'City_Simplified': city_simplified,
        'Duration_Years': request.duration_years,
        'Living_Cost_Index': request.living_cost_index,
        'Rent_USD': request.rent_usd,
        'Visa_Fee_USD': request.visa_fee_usd,
        'Insurance_USD': request.insurance_usd
    }])
    
    return input_data

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "EduSpend TCA Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_name=model_name,
        version="1.0.0"
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    return ModelInfo(
        model_name=model_name,
        version="1.0.0",
        features=feature_names,
        top_cities=top_cities,
        model_performance={
            "r2_score": 0.9644,
            "mae": 2447,
            "rmse": 3930
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_tca(request: PredictionRequest):
    """Predict Total Cost of Attendance for a single request"""
    try:
        if model is not None:
            # Use ML model for prediction
            input_data = prepare_model_input(request)
            prediction = model.predict(input_data)[0]
            confidence_score = 0.95  # Simplified confidence score
        else:
            # Use fallback calculation
            prediction = calculate_tca_fallback(request)
            confidence_score = 0.7  # Lower confidence for fallback
        
        # Calculate cost breakdown
        annual_rent = request.rent_usd * 12
        total_housing = annual_rent * request.duration_years
        estimated_tuition = prediction - total_housing - request.insurance_usd - request.visa_fee_usd
        
        cost_breakdown = {
            "total_housing": total_housing,
            "estimated_tuition": max(estimated_tuition, 0),
            "insurance": request.insurance_usd,
            "visa_fee": request.visa_fee_usd,
            "housing_percentage": (total_housing / prediction * 100) if prediction > 0 else 0,
            "tuition_percentage": (max(estimated_tuition, 0) / prediction * 100) if prediction > 0 else 0
        }
        
        return PredictionResponse(
            predicted_tca=prediction,
            confidence_score=confidence_score,
            cost_breakdown=cost_breakdown,
            model_version="1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """Predict Total Cost of Attendance for multiple requests"""
    try:
        results = []
        for single_request in request.predictions:
            prediction_response = await predict_tca(single_request)
            results.append(prediction_response)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/countries", response_model=List[str])
async def get_available_countries():
    """Get list of available countries"""
    # This would typically come from the dataset
    countries = [
        "USA", "UK", "Canada", "Australia", "Germany", "France", "Netherlands", 
        "Sweden", "India", "China", "Japan", "South Korea", "Italy", "Spain", 
        "Switzerland", "Austria", "Belgium", "Denmark", "Norway", "Finland"
    ]
    return sorted(countries)

@app.get("/programs", response_model=List[str])
async def get_available_programs():
    """Get list of available programs"""
    programs = [
        "Computer Science", "Business", "Engineering", "Medicine", "Law", 
        "Arts", "Science", "Social Sciences", "Mathematics", "Physics", 
        "Economics", "Psychology", "Education", "Architecture", "Design"
    ]
    return sorted(programs)

@app.get("/levels", response_model=List[str])
async def get_available_levels():
    """Get list of available education levels"""
    return ["Bachelors", "Masters", "PhD", "Diploma"]

@app.get("/cities", response_model=List[str])
async def get_top_cities():
    """Get list of top cities"""
    return top_cities

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
