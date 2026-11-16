# Movie Review Sentiment Classifier (NLP)

Classifies movie reviews as positive or negative using the NLTK `movie_reviews` corpus (2,000 reviews, evenly split). The project turns text into TF-IDF features with word pairs included, compares three standard classifiers, and reads back the words each one relies on.

## Results

400 held-out reviews (20% of the corpus). Numbers come from `reports/metrics.json`.

| Model | 5-fold CV accuracy (train) | Test accuracy | Test F1 |
|---|---|---|---|
| Naive Bayes (baseline) | 0.855 | 0.838 | 0.841 |
| Logistic regression | 0.871 | 0.868 | 0.869 |
| Linear SVM | 0.874 | 0.875 | 0.876 |

Logistic regression and the linear SVM are within the cross-validation spread of each other (about 0.012), so I treat them as tied. I kept logistic regression as the saved model because it returns a probability-style score that `predict.py` can show.

Words with the strongest weights (`reports/top_*_terms.csv`): "great", "the best", "perfect", "excellent" on the positive side and "bad", "worst", "boring", "nothing" on the negative side. Some topic words such as "script" and "plot" also lean negative, because reviewers tend to mention them when they are complaining. That is a pattern in this corpus, not a general rule about language.

`reports/errors.csv` lists the 53 misclassified test reviews with a 300-character snippet each, for reading by hand.

## Method

1. **Cleaning** (`sentiment/data.py`): lower-case, remove symbols other than letters, digits, apostrophes, and `!` or `?`, collapse spaces.
2. **Features** (`sentiment/model.py`): TF-IDF with single words and two-word phrases, words must appear in at least 3 reviews, sublinear term frequency.
3. **Models**: Naive Bayes, logistic regression, and linear SVM, each inside one scikit-learn `Pipeline` so the vocabulary is learned from training reviews only.
4. **Evaluation**: stratified 80/20 split, 5-fold cross-validation on the training part, and one final score on the test part.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py                      # downloads the corpus on first run, writes reports/
python predict.py "A slow, dull film with no surprises."
python -m pytest -q                  # 7 tests
```

## Layout

```
sentiment/data.py    corpus loading and text cleaning
sentiment/model.py   pipelines and top-term helper
train.py             trains and compares the models, writes reports/
predict.py           scores new text from the command line
tests/               checks for cleaning and for each model on a tiny corpus
docs/RUNBOOK.md      setup, troubleshooting, how to change things
reports/             metrics, confusion matrix, top terms, errors
```

## Limits

- Small corpus of older, long-form reviews. Short modern text such as tweets or product reviews would likely score worse.
- Bag-of-words features cannot follow sarcasm or negation across a sentence.
- A transformer model would probably score higher. This project is meant as a clear, checked baseline that a team can read and extend.
