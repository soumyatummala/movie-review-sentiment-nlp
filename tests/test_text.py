import pytest

from sentiment.data import clean_text
from sentiment.model import build_pipeline, top_terms


def test_clean_text_lowercases_and_strips_symbols():
    assert clean_text("  GREAT   movie!!! <br> 10/10 ") == "great movie!!! br 10 10"


def test_clean_text_rejects_non_strings():
    with pytest.raises(TypeError):
        clean_text(None)


def tiny_corpus():
    pos = ["a wonderful moving film", "great acting and a great story", "loved every minute, brilliant"] * 6
    neg = ["a boring awful film", "terrible acting and a dull story", "hated every minute, dreadful"] * 6
    return pos + neg, [1] * len(pos) + [0] * len(neg)


@pytest.mark.parametrize("name", ["naive_bayes", "logistic", "linear_svm"])
def test_each_model_learns_a_tiny_corpus(name):
    X, y = tiny_corpus()
    pipe = build_pipeline(name)
    pipe.set_params(tfidf__min_df=1)
    pipe.fit(X, y)
    assert pipe.predict(["a brilliant wonderful film"])[0] == 1
    assert pipe.predict(["an awful dreadful boring film"])[0] == 0


def test_top_terms_returns_expected_sides():
    X, y = tiny_corpus()
    pipe = build_pipeline("logistic")
    pipe.set_params(tfidf__min_df=1)
    pipe.fit(X, y)
    positive, negative = top_terms(pipe, n=5)
    assert len(positive) == 5 and len(negative) == 5
    assert positive[0][1] > 0 > negative[0][1]


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_pipeline("magic")
