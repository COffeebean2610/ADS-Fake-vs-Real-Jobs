"""
train.py - Model Training Script
Fake Job Posting Detection System
Trains Logistic Regression + Random Forest, saves best model via joblib
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from utils.preprocess import clean_text, combine_features

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "fake_job_postings.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
STATIC_DIR  = os.path.join(BASE_DIR, "static", "charts")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# ── 1. Load & Inspect ──────────────────────────────────────────────────────────
def load_data():
    print("[INFO] Loading dataset …")
    df = pd.read_csv(DATA_PATH)
    print("COLUMNS ARE:", df.columns)   # 👈 ADD THIS LINE
    print(f"[INFO] Shape: {df.shape}")
 ##   print(f"[INFO] Fraudulent value counts:\n{df['fraudulent'].value_counts()}")
    return df


def preprocess(df):

    # ✅ 1. Handle missing values (PUT IT HERE)
    df[['job_title','job_description','requirements','company_profile','benefits']] = \
    df[['job_title','job_description','requirements','company_profile','benefits']].fillna("")

    # ✅ 2. Combine text
    df["combined_text"] = df.apply(combine_features, axis=1)

    # ✅ 3. Clean text
    df["combined_text"] = df["combined_text"].apply(clean_text)

    # ✅ 4. Features & target
    X = df["combined_text"]
    y = df["is_fake"]

    return X, y, df


# ── 3. Vectorize ──────────────────────────────────────────────────────────────
def vectorize(X_train, X_test):
    print("[INFO] TF-IDF vectorizing …")
    
    tfidf = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)
    
    return tfidf, X_train_vec, X_test_vec


# ── 4. Train Models ───────────────────────────────────────────────────────────
def train_models(X_train_vec, X_test_vec, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0,
                                                   class_weight="balanced",
                                                   solver="lbfgs"),
        # "Random Forest":       RandomForestClassifier(n_estimators=200,
        #                                               class_weight="balanced",
        #                                               n_jobs=-1,
        #                                               random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"[INFO] Training {name} …")
        model.fit(X_train_vec, y_train)
        y_pred  = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec)[:, 1]

        acc    = accuracy_score(y_test, y_pred)
        roc    = roc_auc_score(y_test, y_proba)
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "model":    model,
            "accuracy": acc,
            "roc_auc":  roc,
            "cm":       cm,
            "report":   report,
        }
        print(f"  Accuracy : {acc:.4f}  |  ROC-AUC : {roc:.4f}")

    return results


# ── 5. Save Charts ────────────────────────────────────────────────────────────
def save_charts(results, y, best_name):
    # ── Confusion matrix ──
    cm = results[best_name]["cm"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)
    print("[INFO] Saved confusion_matrix.png")

    # ── Fake vs Real distribution ──
    counts = y.value_counts().sort_index()
    labels = ["Real (0)", "Fake (1)"]
    colors = ["#4CAF50", "#F44336"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.4, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50, str(val),
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("Fake vs Real Job Distribution")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "distribution.png"), dpi=120)
    plt.close(fig)
    print("[INFO] Saved distribution.png")

    # ── Model accuracy comparison ──
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] * 100 for n in names]
    colors2 = ["#2196F3", "#FF9800"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(names, accs, color=colors2, height=0.4, edgecolor="white")
    for bar, val in zip(bars, accs):
        ax.text(val - 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", ha="right",
                color="white", fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "model_comparison.png"), dpi=120)
    plt.close(fig)
    print("[INFO] Saved model_comparison.png")


# ── 6. Persist artefacts ──────────────────────────────────────────────────────
def save_artifacts(best_model, tfidf, best_name, results, y):
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(tfidf,      os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"[INFO] Model ({best_name}) and vectorizer saved to /models/")

    # Persist analytics metadata as JSON for Flask to read
    total      = len(y)
    fake_count = int(y.sum())
    real_count = total - fake_count
    best_r     = results[best_name]
    cm         = best_r["cm"].tolist()

    meta = {
        "best_model_name": best_name,
        "total_jobs":      total,
        "fake_jobs":       fake_count,
        "real_jobs":       real_count,
        "fake_pct":        round(fake_count / total * 100, 2),
        "real_pct":        round(real_count / total * 100, 2),
        "models": {
            name: {
                "accuracy": round(results[name]["accuracy"] * 100, 2),
                "roc_auc":  round(results[name]["roc_auc"],  4),
                "precision_fake": round(
                    results[name]["report"]["1"]["precision"], 4),
                "recall_fake": round(
                    results[name]["report"]["1"]["recall"],    4),
                "f1_fake": round(
                    results[name]["report"]["1"]["f1-score"],  4),
            }
            for name in results
        },
        "confusion_matrix": cm,
    }

    with open(os.path.join(MODEL_DIR, "analytics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[INFO] analytics.json saved.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df              = load_data()
    X, y, df        = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    tfidf, X_train_vec, X_test_vec = vectorize(X_train, X_test)
    results = train_models(X_train_vec, X_test_vec, y_train, y_test)

    # Pick best model by accuracy
    best_name  = max(results, key=lambda n: results[n]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"\n[INFO] Best model → {best_name}")

    save_charts(results, y, best_name)
    save_artifacts(best_model, tfidf, best_name, results, y)
    print("\n✅  Training complete!")
    print(df['is_fake'].value_counts())
