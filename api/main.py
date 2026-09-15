"""
FastAPI Inference Service for Clinical Trial Disease Category Classification.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.settings import settings
from src.pipeline import clean_text_fast

app = FastAPI(
    title="Clinical Trial Disease Classification API",
    description="Production-grade clinical protocol disease category classification API.",
    version="2.0.0",
)

_model_pipeline = None


def get_pipeline():
    global _model_pipeline
    if _model_pipeline is None:
        path = settings.MODEL_PATH if settings.MODEL_PATH.exists() else settings.FALLBACK_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        _model_pipeline = joblib.load(path)
    return _model_pipeline


class TrialInput(BaseModel):
    summary: str = Field(..., min_length=5, description="Brief summary or eligibility protocol text of the clinical trial")
    top_k: int = Field(default=3, ge=1, le=8, description="Number of top predictions to return")


class CategoryScore(BaseModel):
    category: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    top_predictions: List[CategoryScore]
    inference_time_ms: float


@app.get("/health")
def health() -> dict:
    exists = settings.MODEL_PATH.exists() or settings.FALLBACK_MODEL_PATH.exists()
    return {
        "status": "ok" if exists else "degraded",
        "model_exists": exists,
        "classes": [
            "anxiety", "breast cancer", "chronic obstructive pulmonary disease",
            "covid-19", "glaucoma", "rheumatoid arthritis", "sickle cell anemia",
            "type 2 diabetes"
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_trial(payload: TrialInput) -> PredictionResponse:
    start = time.perf_counter()
    try:
        pipeline = get_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    cleaned = clean_text_fast(payload.summary)
    if not cleaned.strip():
        cleaned = payload.summary.lower()

    probs = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_

    sorted_indices = np.argsort(probs)[::-1]
    top_k = min(payload.top_k, len(classes))

    top_preds = [
        CategoryScore(category=str(classes[idx]), probability=round(float(probs[idx]), 4))
        for idx in sorted_indices[:top_k]
    ]

    best_cat = top_preds[0].category
    best_conf = top_preds[0].probability
    elapsed = (time.perf_counter() - start) * 1000.0

    return PredictionResponse(
        predicted_category=best_cat,
        confidence=best_conf,
        top_predictions=top_preds,
        inference_time_ms=round(elapsed, 2),
    )
