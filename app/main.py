import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional

from app.appeal_guidance import retrieve_appeal_guidance, LLM_HALLUCINATION_MITIGATION_RULES

app = FastAPI(
    title="Claim Outcome Prediction API",
    description="Production API predicting insurance claim outcomes (Paid, Partially_Paid, Denied) and retrieving appeal guidance.",
    version="1.0.0"
)

MODEL_PATH = os.path.join('model', 'claim_pipeline.joblib')
pipeline = None

def get_model_pipeline():
    global pipeline
    if pipeline is None:
        if not os.path.exists(MODEL_PATH):
            # Fallback: train model on the fly if artifact missing
            from src.train import build_and_train_pipeline
            pipeline, _, _ = build_and_train_pipeline()
        else:
            pipeline = joblib.load(MODEL_PATH)
    return pipeline

@app.on_event("startup")
def startup_event():
    get_model_pipeline()

# Custom error handler to enforce HTTP 400 on payload validation failures
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Bad Request",
            "message": "Invalid claim payload. Please check field types and required inputs.",
            "details": exc.errors()
        }
    )

class ClaimRequest(BaseModel):
    payer: str = Field(..., example="Aetna")
    provider: str = Field(..., example="Dr. Rao")
    cpt_code: str = Field(..., example="99213")
    diagnosis_code: str = Field(..., example="M54.5")
    billed_amount: float = Field(..., gt=0, example=220.00)
    patient_age: int = Field(..., ge=0, le=120, example=45)

    @validator('payer', 'provider', 'cpt_code', 'diagnosis_code')
    def non_empty_str(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or blank")
        return v.strip()

class PredictionResponse(BaseModel):
    predicted_status: str
    probabilities: Dict[str, float]

class AppealRequest(BaseModel):
    denial_reason: str = Field(..., example="missing_auth")

@app.get("/")
def health_check():
    model_loaded = (pipeline is not None or os.path.exists(MODEL_PATH))
    return {
        "status": "healthy",
        "service": "Claim Outcome Prediction API",
        "model_loaded": model_loaded
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_claim_outcome(claim: ClaimRequest):
    try:
        model = get_model_pipeline()
        
        input_data = pd.DataFrame([{
            'payer': claim.payer,
            'provider': claim.provider,
            'cpt_code': claim.cpt_code,
            'diagnosis_code': claim.diagnosis_code,
            'billed_amount': claim.billed_amount,
            'patient_age': claim.patient_age
        }])
        
        prediction = model.predict(input_data)[0]
        probabilities_array = model.predict_proba(input_data)[0]
        classes = model.classes_
        
        prob_dict = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probabilities_array)
        }
        
        return PredictionResponse(
            predicted_status=prediction,
            probabilities=prob_dict
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/appeal-guidance")
def get_appeal_guidance(request: AppealRequest):
    result = retrieve_appeal_guidance(request.denial_reason)
    return {
        "denial_reason": request.denial_reason,
        "matched_key": result["matched_reason"],
        "appeal_guidance_snippet": result["snippet"],
        "hallucination_mitigation_rules": LLM_HALLUCINATION_MITIGATION_RULES
    }
