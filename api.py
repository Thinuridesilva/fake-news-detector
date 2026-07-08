"""
FastAPI wrapper that serves both the baseline and DistilBERT models.
Run locally: uvicorn api:app --reload --port 8000
Test: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "your article here"}'
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from preprocessing import clean_text

# ── load baseline model ──────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent / "models"
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
clf = joblib.load(MODEL_DIR / "logreg_model.joblib")

# ── try loading DistilBERT (optional — skipped if not present) ───────────────
distilbert_pipeline = None
BERT_DIR = Path(__file__).resolve().parent / "distilbert_fakenews" / "best_model"
if BERT_DIR.exists():
    from transformers import pipeline as hf_pipeline
    distilbert_pipeline = hf_pipeline(
        "text-classification",
        model=str(BERT_DIR),
        tokenizer=str(BERT_DIR),
        truncation=True,
        max_length=256,
    )

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fake News Detector API",
    description="Classifies news articles as REAL or FAKE using TF-IDF+LogReg and DistilBERT.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str


class ModelPrediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    text_preview: str
    baseline: ModelPrediction
    distilbert: ModelPrediction | None
    agreement: bool | None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "baseline_loaded": True,
        "distilbert_loaded": distilbert_pipeline is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        return {"error": "text field is empty"}

    # baseline prediction
    cleaned = clean_text(req.text)
    X = vectorizer.transform([cleaned])
    proba = clf.predict_proba(X)[0]
    baseline_label = "FAKE" if proba[1] > 0.5 else "REAL"
    baseline_conf = float(round(max(proba), 4))

    # distilbert prediction (if available)
    distilbert_result = None
    agreement = None
    if distilbert_pipeline:
        raw = distilbert_pipeline(req.text[:512])[0]
        # HF returns LABEL_0 / LABEL_1 — map back to REAL / FAKE
        bert_label = "FAKE" if raw["label"] == "LABEL_1" else "REAL"
        bert_conf = float(round(raw["score"], 4))
        distilbert_result = ModelPrediction(label=bert_label, confidence=bert_conf)
        agreement = baseline_label == bert_label

    return PredictResponse(
        text_preview=req.text[:100] + "..." if len(req.text) > 100 else req.text,
        baseline=ModelPrediction(label=baseline_label, confidence=baseline_conf),
        distilbert=distilbert_result,
        agreement=agreement,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
