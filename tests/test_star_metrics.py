import json

import numpy as np

from reviewsense.text_mining.train_sentiment import save_eval, star_metrics


def test_perfect_and_off_by_one():
    labels = np.array([0, 4, 2])                              # 1, 5 and 3 stars
    perfect = np.eye(5)[labels]
    m = star_metrics(labels, perfect)
    assert m["accuracy"] == 1.0 and m["off_by_one"] == 1.0 and m["mae_stars"] == 0.0 and m["macro_f1"] == 1.0
    shifted = np.eye(5)[[1, 3, 2]]                            # 2 of 3 predictions are one star away
    m = star_metrics(labels, shifted)
    assert round(m["accuracy"], 3) == 0.333 and m["off_by_one"] == 1.0 and round(m["mae_stars"], 3) == 0.667


def test_mae_uses_expected_stars():
    # 50/50 between 4 and 5 stars -> expected 4.5; gold 5 stars -> error 0.5 even though argmax picks 4
    m = star_metrics(np.array([4]), np.array([[0, 0, 0, 0.5, 0.5]]))
    assert m["mae_stars"] == 0.5


def test_save_eval_writes_per_example_predictions(tmp_path):
    labels, probs = np.array([0, 1]), np.eye(5)[[0, 2]]
    save_eval(tmp_path / "run", star_metrics(labels, probs), labels, probs)
    saved = json.loads((tmp_path / "run" / "eval.json").read_text())
    assert saved["preds"] == [0, 2] and saved["gold"] == [0, 1] and saved["accuracy"] == 0.5