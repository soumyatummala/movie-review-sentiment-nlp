"""Text classification pipelines."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42


def build_pipeline(name: str) -> Pipeline:
    vec = TfidfVectorizer(
        preprocessor=None,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        stop_words=None,
    )
    if name == "naive_bayes":
        clf = MultinomialNB(alpha=0.5)
    elif name == "logistic":
        clf = LogisticRegression(max_iter=2000, C=4.0, random_state=SEED)
    elif name == "linear_svm":
        clf = LinearSVC(C=0.5, random_state=SEED)
    else:
        raise ValueError(f"unknown model: {name}")
    return Pipeline([("tfidf", vec), ("clf", clf)])


def top_terms(pipe: Pipeline, n: int = 15):
    """Most positive and most negative words from a fitted logistic regression."""
    terms = pipe.named_steps["tfidf"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    order = coefs.argsort()
    negative = [(terms[i], round(float(coefs[i]), 3)) for i in order[:n]]
    positive = [(terms[i], round(float(coefs[i]), 3)) for i in order[::-1][:n]]
    return positive, negative
