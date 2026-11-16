"""Score new text with the trained model.

Run:  python predict.py "A slow, dull film with no surprises."
Needs sentiment_logistic.joblib, which train.py creates.
"""
import sys

import joblib

from sentiment.data import clean_text


def predict(texts, model_path="sentiment_logistic.joblib"):
    pipe = joblib.load(model_path)
    cleaned = [clean_text(t) for t in texts]
    labels = pipe.predict(cleaned)
    scores = pipe.predict_proba(cleaned)[:, 1]
    return [
        {"text": t, "label": "positive" if y == 1 else "negative", "positive_score": round(float(s), 3)}
        for t, y, s in zip(texts, labels, scores)
    ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python predict.py \"some review text\" [more text ...]")
    for row in predict(sys.argv[1:]):
        print(row)
