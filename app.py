"""
Day 15-16 -- Gradio demo for the fake news detector.

Run locally: python app.py
Deploy free: create a new Space on Hugging Face (SDK: Gradio), then push
this file + requirements.txt + the models/ folder to it.
"""

import sys
from pathlib import Path

import gradio as gr
import joblib

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from preprocessing import clean_text  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "models"
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
clf = joblib.load(MODEL_DIR / "logreg_model.joblib")


def predict(text: str) -> str:
    if not text or not text.strip():
        return "Please paste some article text first."
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    proba = clf.predict_proba(X)[0]
    label = "FAKE" if proba[1] > 0.5 else "REAL"
    confidence = max(proba)
    return f"{label} ({confidence:.1%} confidence)"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=8, placeholder="Paste a news article or headline..."),
    outputs="text",
    title="Fake News Detector",
    description="TF-IDF + Logistic Regression baseline. Paste an article and see the prediction.",
)

if __name__ == "__main__":
    demo.launch(share=True)
