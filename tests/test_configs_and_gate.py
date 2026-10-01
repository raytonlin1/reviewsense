import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.significance import compare, passes_gate

CONFIGS = sorted(Path("configs/sentiment").glob("*.yaml"))


@pytest.mark.parametrize("cfg", CONFIGS, ids=[c.stem for c in CONFIGS])
def test_every_sentiment_config_parses(cfg):
    from transformers import TrainingArguments
    from trl import TrlParser

    from reviewsense.text_mining.train_sentiment import TaskArgs

    task, args = TrlParser((TaskArgs, TrainingArguments)).parse_args_and_config(["--config", str(cfg)])
    assert task.export_dir and task.n_test > 0
    if not task.baseline:
        assert args.output_dir != task.export_dir          # checkpoints never mixed into the release artifact


def test_cli_overrides_config():
    from transformers import TrainingArguments
    from trl import TrlParser

    from reviewsense.text_mining.train_sentiment import TaskArgs

    task, args = TrlParser((TaskArgs, TrainingArguments)).parse_args_and_config(
        ["--config", "configs/sentiment/distilbert-20k.yaml", "--learning_rate", "3e-5", "--n_train", "500"])
    assert args.learning_rate == 3e-5 and task.n_train == 500 and args.num_train_epochs == 2


def test_gate_rejects_insignificant_and_accepts_real_improvement():
    rng = np.random.default_rng(1)
    base, small = rng.random(1000) < 0.58, rng.random(1000) < 0.60
    r = compare(base, small)
    assert r["diff_95ci"][0] < 0 < r["diff_95ci"][1] and not passes_gate(r)
    big = base | (rng.random(1000) < 0.25)                     # candidate fixes ~25% of the baseline's errors
    assert passes_gate(compare(base, big))


def test_gate_cli_exit_codes(tmp_path):
    gold = [0] * 400
    for name, correct, g in [("base", 200, gold), ("same", 205, gold), ("better", 320, gold), ("other_test_set", 320, [1] + gold[1:])]:
        (tmp_path / name).mkdir()
        preds = [0] * correct + [1] * (400 - correct)
        (tmp_path / name / "eval.json").write_text(json.dumps({"preds": preds, "gold": g}))
    run = lambda c: subprocess.run([sys.executable, "-m", "experiments.significance", str(tmp_path / "base"), str(tmp_path / c), "--gate"],
                                   capture_output=True, text=True).returncode
    assert run("same") == 1 and run("better") == 0 and run("other_test_set") == 1   # same size, different labels


def test_our_config_fields_never_shadow_library_fields():
    """TaskArgs shares one flat namespace with TrainingArguments. A clash (e.g. our 'data_seed' vs
    TrainingArguments.data_seed) makes the parser fail at startup."""
    from dataclasses import fields

    from transformers import TrainingArguments

    from reviewsense.text_mining.train_sentiment import TaskArgs

    assert not {f.name for f in fields(TaskArgs)} & {f.name for f in fields(TrainingArguments)}
