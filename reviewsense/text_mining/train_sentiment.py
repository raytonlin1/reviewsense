"""Predict Yelp star ratings (1-5) from review text: a TF-IDF baseline or a fine-tuned transformer (HF Trainer).

Every experiment is a YAML file in configs/sentiment/ (reviewed in pull requests); any key can be overridden on the CLI:
  python -m reviewsense.text_mining.train_sentiment --config configs/sentiment/tfidf-baseline.yaml
  python -m reviewsense.text_mining.train_sentiment --config configs/sentiment/distilbert-20k.yaml
  python -m reviewsense.text_mining.train_sentiment --config configs/sentiment/distilbert-20k.yaml --learning_rate 3e-5
  python -m reviewsense.text_mining.train_sentiment --config configs/sentiment/distilbert-20k.yaml --resume true   # after a crash
Then gate the release against the baseline (exit code 1 = not significantly better):
  python -m experiments.significance artifacts/tfidf-baseline artifacts/distilbert-20k --gate

Configuration = TaskArgs (below: data and model choice) + transformers.TrainingArguments (all ~100 training
hyperparameters, documented by Hugging Face). Data: the train split is divided into train / validation (model
selection); the official test split is scored exactly once.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score

from ..data import load_hf_yelp
from ..runinfo import write_run_record

log = logging.getLogger(__name__)
LABELS = {i: f"{i + 1} star{'s' if i else ''}" for i in range(5)}      # same label names as the nlptown model
DATASET = "Yelp/yelp_review_full"


@dataclass
class TaskArgs:
    model_name_or_path: str = "bert-base-uncased"
    export_dir: str = "artifacts/sentiment-model"   # the release artifact (model + card + run.json + eval.json);
                                                    # TrainingArguments.output_dir holds resumable checkpoints only
    baseline: bool = False                     # TF-IDF + logistic regression instead of a transformer
    n_train: int = 20000                       # sampled from the train split, then divided into train / validation
    n_test: int = 2000                         # sampled from the official test split
    val_fraction: float = 0.1
    max_length: int = 256                      # tokens; attention cost grows with the square of this
    split_seed: int = 42                       # train/validation split (must not clash with TrainingArguments.data_seed)
    tfidf_max_features: int = 200_000          # baseline only
    logreg_c: float = 4.0                      # baseline only (inverse regularisation strength)
    resume: bool = False                       # continue from the latest checkpoint in output_dir after a crash


def star_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """labels 0..4, probs (n, 5). Off-by-one: 4 vs 5 stars is a small error. MAE uses the expected star value."""
    preds = probs.argmax(-1)
    expected = (probs * np.arange(1, 6)).sum(-1)
    return {"accuracy": float((preds == labels).mean()),
            "off_by_one": float((np.abs(preds - labels) <= 1).mean()),
            "mae_stars": float(np.abs(expected - (labels + 1)).mean()),
            "macro_f1": float(f1_score(labels, preds, average="macro"))}


def splits(task: TaskArgs):
    train = load_hf_yelp("train", task.n_train).train_test_split(test_size=task.val_fraction, seed=task.split_seed)
    return train["train"], train["test"], load_hf_yelp("test", task.n_test)


def run_baseline(task: TaskArgs, args) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    train, _, test = splits(task)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=task.tfidf_max_features),
                          LogisticRegression(max_iter=2000, C=task.logreg_c))
    model.fit(train["text"], train["label"])
    probs, labels = model.predict_proba(test["text"]), np.array(test["label"])
    metrics = star_metrics(labels, probs)
    write_run_record(task.export_dir, {"task": task}, metrics, labels, probs.argmax(-1))
    return metrics


def run_transformer(task: TaskArgs, args) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

    tok = AutoTokenizer.from_pretrained(task.model_name_or_path)
    enc = lambda b: tok(b["text"], truncation=True, max_length=task.max_length)
    train, val, test = (d.map(enc, batched=True, remove_columns=["text"]) for d in splits(task))
    model = AutoModelForSequenceClassification.from_pretrained(
        task.model_name_or_path, num_labels=5, id2label=LABELS, label2id={v: k for k, v in LABELS.items()})
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, processing_class=tok,
                      data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=lambda p: star_metrics(p.label_ids, softmax(p.predictions, axis=-1)))
    trainer.train(resume_from_checkpoint=task.resume)
    trainer.save_model(task.export_dir)                                     # best checkpoint (validation) -> export
    trainer.args.output_dir = task.export_dir                               # model card is written to output_dir
    trainer.create_model_card(model_name=Path(task.export_dir).name, finetuned_from=task.model_name_or_path,
                              tasks="text-classification", dataset=DATASET)
    pred = trainer.predict(test, metric_key_prefix="test")                  # the one and only look at the test set
    labels, probs = pred.label_ids, softmax(pred.predictions, axis=-1)
    metrics = star_metrics(labels, probs)
    write_run_record(task.export_dir, {"task": task, "training_args": args}, metrics, labels, probs.argmax(-1))
    return metrics


def main(argv: list[str] | None = None) -> dict:
    import torch
    from transformers import TrainingArguments
    from trl import TrlParser

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not torch.cuda.is_available():
        argv += ["--bf16", "false"]          # bf16 only pays off on recent NVIDIA GPUs; parsed so it applies at construction
    task, args = TrlParser((TaskArgs, TrainingArguments)).parse_args_and_config(argv)
    metrics = run_baseline(task, args) if task.baseline else run_transformer(task, args)
    log.info("test metrics %s -> %s", json.dumps({k: round(v, 4) for k, v in metrics.items()}), task.export_dir)
    return metrics


if __name__ == "__main__":
    main()
