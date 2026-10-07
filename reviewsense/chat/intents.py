"""Intent classification: what does the user want to do? Few-shot, with SetFit.

SetFit fine-tunes a sentence-embedding model (Part 7's all-MiniLM-L6-v2) on pairs of examples ("same intent?") and
trains a small classifier on top: a few labelled examples per intent are enough, and training takes seconds on a CPU.
Restaurant names are replaced by "the restaurant" before training and classifying (delexicalization), so the model
learns the request ("book a table at ...") rather than which restaurants appear in the examples.

  python -m reviewsense.chat.intents          # trains on data/intents.yaml, saves to settings.intent_model
"""
from __future__ import annotations

import argparse
from functools import cache

import yaml
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from ..config import get_settings
from ..linguistics.core import nlp


def delexicalize(text: str) -> str:
    """"Book a table at Golden Dragon" -> "Book a table at the restaurant" (our businesses only, via Part 5's rules)."""
    doc = nlp()(text)
    parts, last = [], 0
    for entity in doc.ents:
        if entity.label_ == "ORG" and entity.ent_id_:
            parts.append(text[last:entity.start_char] + "the restaurant")
            last = entity.end_char
    return "".join(parts) + text[last:]


def load_examples(name: str) -> tuple[list[str], list[str]]:
    data = yaml.safe_load((get_settings().data_dir / name).read_text())
    texts, labels = [], []
    for intent, examples in data.items():
        for example in examples:
            texts.append(delexicalize(example))
            labels.append(intent)
    return texts, labels


def train(out_dir: str, base_model: str, epochs: int, iterations: int, seed: int) -> SetFitModel:
    texts, labels = load_examples("intents.yaml")
    model = SetFitModel.from_pretrained(base_model, labels=sorted(set(labels)))
    # No checkpoints: training takes seconds, and SetFit's default writes them into ./checkpoints (the project root).
    args = TrainingArguments(num_epochs=epochs, batch_size=16, num_iterations=iterations, seed=seed,
                             save_strategy="no", output_dir=str(get_settings().artifacts_dir / "checkpoints" / "intents"))
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_dict({"text": texts, "label": labels}))
    trainer.train()
    model.save_pretrained(out_dir)
    return model


@cache
def load_intent_model() -> SetFitModel:
    return SetFitModel.from_pretrained(get_settings().intent_model)


def classify(text: str) -> tuple[str, float]:
    """-> (intent, confidence 0-1)."""
    model = load_intent_model()
    probabilities = model.predict_proba([delexicalize(text)])[0]
    best = int(probabilities.argmax())
    return model.labels[best], round(float(probabilities[best]), 3)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the intent classifier (SetFit).")
    parser.add_argument("--out_dir", default=get_settings().intent_model)
    parser.add_argument("--base_model", default=get_settings().embedding_model)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20, help="example pairs generated per example")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    train(args.out_dir, args.base_model, args.epochs, args.iterations, args.seed)
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
