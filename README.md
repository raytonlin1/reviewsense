# ReviewSense

A conversational NLP system for customer reviews, built part by part. This repository currently contains
**Parts 0–2**: project setup, text preprocessing, and star-rating prediction (TF-IDF baseline vs. fine-tuned BERT)
with config-driven training, run records and a statistical release gate.

## Setup (Python 3.14)
```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python -m spacy download en_core_web_sm
python -m pytest
```

## Train and gate a model
```bash
python -m reviewsense.text_mining.train_sentiment --config configs/sentiment/tfidf-baseline.yaml
python -m reviewsense.text_mining.train_sentiment --config configs/sentiment/distilbert-20k.yaml
python -m experiments.significance artifacts/tfidf-baseline artifacts/distilbert-20k --gate
```
Every run writes `export_dir/` with the model, a model card, `run.json` (configs, metrics, git commit, library
versions) and `eval.json` (per-review predictions for the gate). Any config key can be overridden on the command
line, e.g. `--learning_rate 3e-5`.

## Layout
| Path | Part |
|---|---|
| `reviewsense/config.py`, `reviewsense/__init__.py` | 0: settings from environment variables |
| `reviewsense/data.py` | 0–1: loaders, cleaning (ftfy, BeautifulSoup), sentence splitting (spaCy senter) |
| `reviewsense/text_mining/train_sentiment.py`, `configs/sentiment/` | 2: TF-IDF baseline and BERT fine-tuning |
| `reviewsense/runinfo.py`, `experiments/significance.py` | 2: run records and the release gate |
