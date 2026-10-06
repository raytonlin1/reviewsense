"""Run records: everything needed to trace a trained model back to how it was made (lineage / reproducibility).

Written next to every model: run.json (configs, metrics, git commit, library versions) and eval.json (the
per-review predictions the release gate compares). Teams with an experiment tracker (MLflow, Weights & Biases;
`report_to: mlflow` in the training config) get most of run.json from it; eval.json is still needed for the gate.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

PACKAGES = ["torch", "transformers", "trl", "datasets", "scikit-learn"]


def git_commit() -> str | None:
    """The code version, e.g. "0926e51" or "0926e51-dirty" (uncommitted changes: the record is not reproducible)."""
    try:
        result = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=5)
    except OSError:
        return None
    return result.stdout.strip() or None


def as_dict(config) -> dict:
    """Our dataclasses -> dict; Hugging Face's TrainingArguments has its own to_dict()."""
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return asdict(config)


def write_run_record(out_dir: str | Path, configs: dict, metrics: dict,
                     labels: np.ndarray | None = None, preds: np.ndarray | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {package: version(package) for package in PACKAGES},
        "configs": {name: as_dict(config) for name, config in configs.items()},
        "metrics": metrics,
    }
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str))
    if labels is not None and preds is not None:                # classifiers: per-example predictions for the gate
        (out / "eval.json").write_text(json.dumps({**metrics, "preds": preds.tolist(), "gold": labels.tolist()}))
