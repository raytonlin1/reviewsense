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
from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score

from ..data import load_hf_yelp
from ..runinfo import write_run_record

log = logging.getLogger(__name__)
# The model predicts a class number 0-4. These names are saved with the model so its predictions read "4 stars",
# not "LABEL_3" (they also match the public nlptown model the app uses until you train your own).
ID2LABEL = {0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"}
LABEL2ID = {name: number for number, name in ID2LABEL.items()}


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


def compute_metrics(eval_pred) -> dict:
    """The Trainer calls this after every evaluation: raw model scores (logits) -> probabilities -> our metrics."""
    logits, labels = eval_pred
    return star_metrics(labels, softmax(logits, axis=-1))


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
    """Fine-tune a pretrained transformer (e.g. BERT) to predict 1-5 stars. `args` holds the training settings
    (learning rate, epochs, batch size...) read from the YAML config."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

    # 1. Data: train (learn), validation (pick the best checkpoint) and test (final score, used once).
    train, validation, test = splits(task)

    # 2. Tokenize: turn each review into the token IDs the model reads, cut to max_length tokens.
    tokenizer = AutoTokenizer.from_pretrained(task.model_name_or_path)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=task.max_length)

    train = train.map(tokenize, batched=True)
    validation = validation.map(tokenize, batched=True)
    test = test.map(tokenize, batched=True)

    # 3. Model: the pretrained transformer plus a new, untrained layer that outputs 5 scores (one per star).
    model = AutoModelForSequenceClassification.from_pretrained(
        task.model_name_or_path, num_labels=5, id2label=ID2LABEL, label2id=LABEL2ID)

    # 4. Trainer: the library runs the whole training loop; we only plug in the pieces.
    trainer = Trainer(
        model=model,
        args=args,                                         # all training settings, from the YAML config
        train_dataset=train,
        eval_dataset=validation,                           # evaluated each epoch; the best epoch is kept
        processing_class=tokenizer,                        # saved next to the model so it can be reloaded
        data_collator=DataCollatorWithPadding(tokenizer),  # pads each batch only to its longest review
        compute_metrics=compute_metrics,
    )

    # 5. Train (resume=true continues a crashed run from its last checkpoint), then save the best model.
    trainer.train(resume_from_checkpoint=task.resume)
    trainer.save_model(task.export_dir)

    # 6. Score the test set exactly once, and record how this model was made.
    output = trainer.predict(test)
    labels, probs = output.label_ids, softmax(output.predictions, axis=-1)
    metrics = star_metrics(labels, probs)
    write_run_record(task.export_dir, {"task": task, "training_args": args}, metrics, labels, probs.argmax(-1))
    return metrics


def main(argv: list[str] | None = None) -> dict:
    from transformers import TrainingArguments
    from trl import TrlParser

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Read the YAML config (plus any --overrides) into our TaskArgs and Hugging Face's TrainingArguments.
    task, args = TrlParser((TaskArgs, TrainingArguments)).parse_args_and_config(argv)
    metrics = run_baseline(task, args) if task.baseline else run_transformer(task, args)
    log.info("test metrics %s -> %s", json.dumps({k: round(v, 4) for k, v in metrics.items()}), task.export_dir)
    return metrics


if __name__ == "__main__":
    main()
