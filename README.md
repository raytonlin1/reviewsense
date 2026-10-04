# ReviewSense

A conversational NLP system for customer reviews, built part by part with the libraries companies use
(Hugging Face, spaCy, scikit-learn, Haystack). This repository currently contains **Parts 0–7**.

## Setup (Python 3.14)
```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python -m spacy download en_core_web_sm
python -m pytest                  # fast tests (seconds)
python -m pytest -m integration   # tests that download and run real models (minutes)
```

## Train and gate a model (Part 2)
```bash
python -m reviewsense.text_mining.train_sentiment --config configs/sentiment/tfidf-baseline.yaml
python -m reviewsense.text_mining.train_sentiment --config configs/sentiment/distilbert-20k.yaml
python -m experiments.significance artifacts/tfidf-baseline artifacts/distilbert-20k --gate
```
Every run writes `export_dir/` with the model, `run.json` (configs, metrics, git commit, library versions) and
`eval.json` (per-review predictions for the gate). Any config key can be overridden on the command line,
e.g. `--learning_rate 3e-5`.

## Layout
| Path | Part | Demo |
|---|---|---|
| `reviewsense/config.py`, `reviewsense/__init__.py` | 0: settings from environment variables / `.env` | |
| `reviewsense/data.py` | 0–1: loaders, cleaning (ftfy, BeautifulSoup), sentence splitting (spaCy) | `test.py` |
| `reviewsense/text_mining/train_sentiment.py`, `configs/sentiment/` | 2: TF-IDF baseline and BERT fine-tuning | |
| `reviewsense/runinfo.py`, `experiments/significance.py` | 2: run records and the release gate | |
| `reviewsense/text_mining/analyzer.py` | 3: stars, emotions, aspect sentiment per business | `analyze_demo.py` |
| `reviewsense/text_mining/topics.py` | 4: topic discovery (TF-IDF + NMF) | `topics_demo.py` |
| `reviewsense/linguistics/` | 5: shared spaCy pipeline with entity rules, NER evaluation | `linguistics_demo.py` |
| `reviewsense/kg/` | 6: fact extraction into a knowledge graph, question answering | `kg_demo.py` |
| `reviewsense/search/` | 7: hybrid search (BM25 + embeddings + reranker), spelling, evaluation | `search_demo.py` |
