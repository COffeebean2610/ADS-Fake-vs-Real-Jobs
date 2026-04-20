"""
utils/model_loader.py
Loads the trained model and TF-IDF vectorizer ONCE at app startup.
All Flask routes import from this module — no redundant disk reads.
"""

import os
import json
import joblib

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

_model     = None
_vectorizer = None
_analytics  = None


def load_model():
    global _model, _vectorizer
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    vec_path   = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Run  python train.py  first."
        )

    _model      = joblib.load(model_path)
    _vectorizer = joblib.load(vec_path)
    print("[INFO] Model and vectorizer loaded.")


def get_model():
    if _model is None:
        load_model()
    return _model, _vectorizer


def get_analytics():
    global _analytics
    if _analytics is None:
        analytics_path = os.path.join(MODEL_DIR, "analytics.json")
        if os.path.exists(analytics_path):
            with open(analytics_path) as f:
                _analytics = json.load(f)
        else:
            _analytics = {}
    return _analytics
