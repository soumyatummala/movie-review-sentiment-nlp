"""Train and compare the sentiment models.

Run:  python train.py
Writes metrics, charts, and a list of misclassified reviews into reports/.
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from sentiment.data import clean_text, load_reviews
from sentiment.model import SEED, build_pipeline, top_terms

REPORTS = Path("reports")


def main():
    REPORTS.mkdir(exist_ok=True)
    df = load_reviews()
    df["clean"] = df["text"].map(clean_text)

    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name in ["naive_bayes", "logistic", "linear_svm"]:
        pipe = build_pipeline(name)
        cv_acc = cross_val_score(pipe, train["clean"], train["label"], cv=cv, scoring="accuracy")
        pipe.fit(train["clean"], train["label"])
        pred = pipe.predict(test["clean"])
        results[name] = {
            "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
            "cv_accuracy_std": round(float(cv_acc.std()), 4),
            "test_accuracy": round(float(accuracy_score(test["label"], pred)), 4),
            "test_f1": round(float(f1_score(test["label"], pred)), 4),
        }
        fitted[name] = (pipe, pred)

    results["rows"] = {"train": len(train), "test": len(test)}
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))

    # Words that push a review toward positive or negative
    pos, neg = top_terms(fitted["logistic"][0], n=15)
    pd.DataFrame({"positive_term": [t for t, _ in pos], "weight": [w for _, w in pos]}).to_csv(
        REPORTS / "top_positive_terms.csv", index=False
    )
    pd.DataFrame({"negative_term": [t for t, _ in neg], "weight": [w for _, w in neg]}).to_csv(
        REPORTS / "top_negative_terms.csv", index=False
    )

    # Confusion matrix for the logistic model
    pipe, pred = fitted["logistic"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        test["label"], pred, display_labels=["negative", "positive"], ax=ax
    )
    ax.set_title("Logistic regression on test reviews")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix_logistic.png", dpi=150)
    plt.close(fig)

    # Misclassified reviews, for reading by hand
    wrong = test.assign(predicted=pred)
    wrong = wrong[wrong["label"] != wrong["predicted"]].copy()
    wrong["snippet"] = wrong["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 300)
    wrong[["file", "label", "predicted", "snippet"]].to_csv(REPORTS / "errors.csv", index=False)

    joblib.dump(pipe, "sentiment_logistic.joblib")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
