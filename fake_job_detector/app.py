"""
app.py – Fake Job Posting Detection System
Flask multi-page web application
"""

from flask import Flask, render_template, request, redirect, url_for, session
from utils.preprocess import clean_text, combine_features
from utils.model_loader import get_model, get_analytics, load_model

app = Flask(__name__)
app.secret_key = "fjd_secret_2024"   # for session flash messages

# Load model once at startup
load_model()


# ── HOME / DASHBOARD ──────────────────────────────────────────────────────────
@app.route("/")
def home():
    analytics = get_analytics()
    return render_template("index.html", analytics=analytics)


# ── PREDICTION FORM ───────────────────────────────────────────────────────────
@app.route("/predict", methods=["GET"])
def predict():
    return render_template("predict.html")


# ── PREDICTION RESULT (POST) ──────────────────────────────────────────────────
@app.route("/result", methods=["POST"])
def result():
    title       = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    requirements = request.form.get("requirements", "").strip()
    company     = request.form.get("company", "").strip()
    benefits    = request.form.get("benefits", "").strip()

    # Build combined text exactly as during training
    row = {
    "job_title":       title,
    "company_profile": company,
    "job_description": description,
    "requirements":    requirements,
    "benefits":        benefits,
}
    combined  = combine_features(row)
    cleaned   = clean_text(combined)

    model, vectorizer = get_model()
    vec        = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    # proba      = model.predict_proba(vec)[0]

    # label      = "Fake Job"    if prediction == 1 else "Real Job"
    # confidence = round(proba[prediction] * 100, 2)
    # fake_prob  = round(proba[1] * 100, 2)
    # real_prob  = round(proba[0] * 100, 2)
    proba = model.predict_proba(vec)[0]

    fake_prob_value = proba[1]

    if fake_prob_value > 0.7:
        prediction = 1
        label = "Fake Job"
        confidence = round(fake_prob_value * 100, 2)
    else:
        prediction = 0
        label = "Real Job"
        confidence = round(proba[0] * 100, 2)

    fake_prob  = round(proba[1] * 100, 2)
    real_prob  = round(proba[0] * 100, 2)
    context = {
        "label":       label,
        "confidence":  confidence,
        "fake_prob":   fake_prob,
        "real_prob":   real_prob,
        "is_fake":     prediction == 1,
        "input": {
            "title":        title,
            "company":      company,
            "description":  description,
            "requirements": requirements,
            "benefits":     benefits,
        }
    }
    return render_template("result.html", **context)


# ── ANALYTICS PAGE ────────────────────────────────────────────────────────────
@app.route("/analytics")
def analytics():
    data = get_analytics()
    return render_template("analytics.html", data=data)


# ── ABOUT PAGE ────────────────────────────────────────────────────────────────
@app.route("/about")
def about():
    return render_template("about.html")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
