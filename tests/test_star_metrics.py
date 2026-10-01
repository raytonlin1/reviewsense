import json

import numpy as np

from reviewsense.runinfo import write_run_record
from reviewsense.text_mining.train_sentiment import TaskArgs, star_metrics


def test_perfect_and_off_by_one():
    labels = np.array([0, 4, 2])                              # 1, 5 and 3 stars
    m = star_metrics(labels, np.eye(5)[labels])
    assert m["accuracy"] == 1.0 and m["off_by_one"] == 1.0 and m["mae_stars"] == 0.0 and m["macro_f1"] == 1.0
    m = star_metrics(labels, np.eye(5)[[1, 3, 2]])            # 2 of 3 predictions are one star away
    assert round(m["accuracy"], 3) == 0.333 and m["off_by_one"] == 1.0 and round(m["mae_stars"], 3) == 0.667


def test_mae_uses_expected_stars():
    # 50/50 between 4 and 5 stars -> expected 4.5; gold 5 stars -> error 0.5 even though argmax picks 4
    assert star_metrics(np.array([4]), np.array([[0, 0, 0, 0.5, 0.5]]))["mae_stars"] == 0.5


def test_run_record_has_lineage_and_predictions(tmp_path):
    labels, preds = np.array([0, 1]), np.array([0, 2])
    write_run_record(tmp_path / "run", {"task": TaskArgs(n_train=10)}, {"accuracy": 0.5}, labels, preds)
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["configs"]["task"]["n_train"] == 10 and run["packages"]["transformers"] and run["metrics"]["accuracy"] == 0.5
    ev = json.loads((tmp_path / "run" / "eval.json").read_text())
    assert ev["preds"] == [0, 2] and ev["gold"] == [0, 1]
