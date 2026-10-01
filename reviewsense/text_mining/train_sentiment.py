#script with two models:
#A baseline: TF-IDF + logistic regression (scikit-learn), which trains in seconds.
#Fine-tuned BERT: trained with the Hugging Face Trainer, on a proper train / validation / test split.

"""Predict Yelp star ratings (1-5) from review text: a TF-IDF baseline, then a fine-tuned BERT (Hugging Face Trainer).

  python -m reviewsense.text_mining.train_sentiment --baseline --n_train 20000          # seconds, any machine
  python -m reviewsense.text_mining.train_sentiment --n_train 20000 --epochs 2          # BERT; GPU recommended
  python -m reviewsense.text_mining.train_sentiment --model distilbert-base-uncased --n_train 2000 --epochs 1   # CPU smoke test
  python -m experiments.significance artifacts/tfidf-yelp/eval.json artifacts/bert-yelp/eval.json

Data: train split -> 90% train / 10% validation (model selection), the official test split is scored ONCE at the end.
Afterwards point the app at the model:  REVIEWSENSE_SENTIMENT_MODEL=artifacts/bert-yelp
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score

from ..config import get_settings
from ..data import load_hf_yelp

LABELS = {i: f"{i + 1} star{'s' if i else ''}" for i in range(5)}      # same label names as the nlptown model


def star_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """labels 0..4, probs (n, 5). Off-by-one: 4 vs 5 stars is a small error. MAE uses the expected star value."""
    preds = probs.argmax(-1)
    expected = (probs * np.arange(1, 6)).sum(-1)
    return {"accuracy": float((preds == labels).mean()),
            "off_by_one": float((np.abs(preds - labels) <= 1).mean()),
            "mae_stars": float(np.abs(expected - (labels + 1)).mean()),
            "macro_f1": float(f1_score(labels, preds, average="macro"))}


def save_eval(out_dir: str | Path, metrics: dict, labels: np.ndarray, probs: np.ndarray) -> None:
    """Per-example predictions next to the model, so any two runs can be compared with experiments/significance.py."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps({**metrics, "preds": probs.argmax(-1).tolist(), "gold": labels.tolist()}))


def splits(n_train: int, n_test: int, seed: int = 42):
    train = load_hf_yelp("train", n_train).train_test_split(test_size=0.1, seed=seed)
    return train["train"], train["test"], load_hf_yelp("test", n_test)


def run_baseline(args) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    train, _, test = splits(args.n_train, args.n_eval)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
                          LogisticRegression(max_iter=2000, C=4.0))
    model.fit(train["text"], train["label"])
    probs, labels = model.predict_proba(test["text"]), np.array(test["label"])
    metrics = star_metrics(labels, probs)
    save_eval(args.out, metrics, labels, probs)
    return metrics


def run_bert(args) -> dict:
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda b: tok(b["text"], truncation=True, max_length=args.max_len)
    train, val, test = (d.map(enc, batched=True, remove_columns=["text"]) for d in splits(args.n_train, args.n_eval))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=5, id2label=LABELS, label2id={v: k for k, v in LABELS.items()})
    cuda = torch.cuda.is_available()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(Path(args.out) / "checkpoints"), learning_rate=args.lr, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs * 2,
            warmup_steps=0.06, weight_decay=0.01, lr_scheduler_type="linear",
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model="macro_f1",       # selected on VALIDATION, not test
            bf16=cuda and torch.cuda.is_bf16_supported(), use_cpu=args.cpu,
            logging_steps=50, report_to="none", seed=42),
        train_dataset=train, eval_dataset=val, processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=lambda p: star_metrics(p.label_ids, softmax(p.predictions, axis=-1)))
    trainer.train()
    trainer.save_model(args.out)
    shutil.rmtree(Path(args.out) / "checkpoints", ignore_errors=True)    # optimizer state: only needed to resume
    pred = trainer.predict(test)                                            # the one and only look at the test set
    probs, labels = softmax(pred.predictions, axis=-1), pred.label_ids
    metrics = star_metrics(labels, probs)
    save_eval(args.out, metrics, labels, probs)
    return metrics


def main():
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="TF-IDF + logistic regression instead of BERT")
    ap.add_argument("--model", default=s.sentiment_base)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_eval", type=int, default=2000, help="examples from the official test split")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--cpu", action="store_true", help="force CPU (default: CUDA, else Apple MPS, else CPU)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or str(s.artifacts_dir / ("tfidf-yelp" if args.baseline else "bert-yelp"))
    metrics = run_baseline(args) if args.baseline else run_bert(args)
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}))


if __name__ == "__main__":
    main()
