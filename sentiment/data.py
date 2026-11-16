"""Loading the NLTK movie review corpus (1,000 positive and 1,000 negative reviews)."""
import re
from pathlib import Path

import nltk
import pandas as pd

NLTK_DIR = Path("nltk_data")


def load_reviews() -> pd.DataFrame:
    """Return a DataFrame with columns: text, label (1 = positive, 0 = negative)."""
    NLTK_DIR.mkdir(exist_ok=True)
    nltk.data.path.insert(0, str(NLTK_DIR))
    try:
        from nltk.corpus import movie_reviews

        movie_reviews.fileids()
    except LookupError:
        nltk.download("movie_reviews", download_dir=str(NLTK_DIR), quiet=True)
        from nltk.corpus import movie_reviews

    rows = []
    for fid in movie_reviews.fileids():
        rows.append(
            {
                "file": fid,
                "text": movie_reviews.raw(fid),
                "label": 1 if fid.startswith("pos/") else 0,
            }
        )
    return pd.DataFrame(rows)


def clean_text(text: str) -> str:
    """Light cleanup: lower-case, collapse whitespace, keep words and basic punctuation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.lower()
    text = re.sub(r"[^a-z0-9'!? ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
