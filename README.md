# 🛡️ JobGuard AI — Fake Job Posting Detection System

An NLP-powered multi-page web application that detects fraudulent job postings using Machine Learning.
Built with **Flask**, **Scikit-learn**, **Bootstrap 5**, and **Chart.js**.

---

## 📁 Project Structure

```
fake_job_detector/
│
├── app.py                  # Flask application (routes)
├── train.py                # ML training script
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py       # NLP text cleaning
│   └── model_loader.py     # Loads model once at startup
│
├── models/                 # Auto-created after training
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── analytics.json
│
├── data/
│   └── fake_job_postings.csv   ← place Kaggle dataset here
│
├── templates/
│   ├── base.html           # Shared navbar + layout
│   ├── index.html          # Home / Dashboard
│   ├── predict.html        # Prediction form
│   ├── result.html         # Prediction result
│   ├── analytics.html      # Model metrics & charts
│   └── about.html          # Project info
│
├── static/
│   ├── css/style.css
│   └── charts/             # Auto-created chart images
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone / Download the Project

```bash
git clone <your-repo-url>
cd fake_job_detector
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

- Go to: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
- Download `fake_job_postings.csv`
- Place it inside the `data/` folder:

```
data/
└── fake_job_postings.csv
```

### 5. Train the Model

```bash
python train.py
```

This will:
- Preprocess and vectorize the dataset
- Train Logistic Regression + Random Forest
- Save the best model to `models/best_model.pkl`
- Save charts to `static/charts/`
- Save analytics metadata to `models/analytics.json`

### 6. Run the Flask App

```bash
python app.py
```

Open your browser at → **http://localhost:5000**

---

## 🌐 Pages

| Route         | Page            | Description                          |
|---------------|-----------------|--------------------------------------|
| `/`           | Dashboard       | Stats, dataset overview, how it works|
| `/predict`    | Detect          | Enter job details for analysis       |
| `/result`     | Result          | Verdict + confidence score + charts  |
| `/analytics`  | Analytics       | Model metrics, confusion matrix      |
| `/about`      | About           | Project overview, tech stack         |

---

## 🤖 Machine Learning Pipeline

```
Raw CSV → Fill NaN → Combine text columns
       → clean_text() (NLP preprocessing)
       → TF-IDF Vectorizer (10k features, bigrams)
       → Logistic Regression | Random Forest
       → Compare accuracy → Save best model
```

**NLP Steps in `clean_text()`:**
1. Lowercase
2. Remove URLs and HTML tags
3. Remove non-alphabetic characters
4. Remove NLTK English stop words
5. Filter tokens shorter than 3 chars

---

## 📊 Model Performance (example)

| Model                | Accuracy | ROC-AUC |
|----------------------|----------|---------|
| Logistic Regression  | ~98.2%   | ~0.98   |
| Random Forest        | ~97.8%   | ~0.97   |

> Actual results will vary slightly based on random seed.

---

## 🛠️ Tech Stack

| Layer        | Technology                   |
|--------------|------------------------------|
| Backend      | Python 3.11, Flask 3         |
| ML / NLP     | Scikit-learn, NLTK, Joblib   |
| Data         | Pandas, NumPy                |
| Charts       | Matplotlib, Seaborn, Chart.js|
| Frontend     | HTML5, CSS3, Bootstrap 5     |
| Fonts        | Space Grotesk, DM Mono       |

---

## 👨‍💻 Author

Final Year B.E. (Computer Engineering) Project  
Built with ❤️ using Flask + Scikit-learn
