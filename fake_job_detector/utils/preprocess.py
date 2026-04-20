"""
utils/preprocess.py
NLP preprocessing helpers shared by train.py and app.py
"""

import re
import string
import nltk

# Download NLTK data if not already present
for pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Lowercase → remove URLs → remove punctuation/digits
    → remove stopwords → strip extra whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                    # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                 # non-alpha
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def combine_features(row) -> str:
    cols = [
        "job_title",
        "company_profile",
        "job_description",
        "requirements",
        "benefits"
    ]
    
    parts = []
    for col in cols:
        val = row.get(col, "") if isinstance(row, dict) else getattr(row, col, "")
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    
    return " ".join(parts)