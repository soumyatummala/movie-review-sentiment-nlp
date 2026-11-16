# Runbook: Movie review sentiment project

## Setup
1. Python 3.10 or newer.
2. `pip install -r requirements.txt` inside a virtual environment.
3. `python train.py`. The first run downloads the NLTK `movie_reviews` corpus into `nltk_data/`. After that it works offline.

## Expected output
- `reports/metrics.json` with cross-validation and test scores for three models.
- `reports/confusion_matrix_logistic.png`, `top_positive_terms.csv`, `top_negative_terms.csv`, and `errors.csv`.
- `sentiment_logistic.joblib`, the fitted logistic pipeline used by `predict.py` (not committed).
- Results are repeatable because every random step uses seed 42.

## Troubleshooting
| Symptom | Likely cause | Fix |
|---|---|---|
| `LookupError` or download failure for `movie_reviews` | No internet on first run | Run `python -m nltk.downloader -d nltk_data movie_reviews` on a connected machine and copy `nltk_data/` over |
| `FileNotFoundError: sentiment_logistic.joblib` | `predict.py` run before training | Run `python train.py` first |
| `TypeError: text must be a string` | Empty or missing text passed in | Filter out missing values before scoring |
| Scores differ slightly | Different scikit-learn version | Small differences are normal; keep the same seed |

## Changing things
- New model: add a branch in `build_pipeline`, add the name to the list in `train.py`, and add it to the parametrized test in `tests/test_text.py`.
- New data: replace `load_reviews` so it returns a DataFrame with `text` and `label` columns. The rest of the code stays the same.
- After any change, run `python -m pytest -q` before `python train.py`.

## Data privacy
The corpus is public movie reviews. If you adapt this to customer text, remove names and contact details first, keep raw text out of git, and follow your organization's rules for data access and responsible AI use.
