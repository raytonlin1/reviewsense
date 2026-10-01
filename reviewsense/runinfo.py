"""Run records: everything needed to trace a trained model back to how it was made (lineage / reproducibility).
Written next to every model as run.json; per-example predictions go to eval.json for release gating."""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

PACKAGES = ("torch", "transformers", "trl", "datasets", "scikit-learn")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict[str, str | None]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def _plain(cfg):
    if hasattr(cfg, "to_dict"):           # transformers / TRL argument dataclasses
        return cfg.to_dict()
    return asdict(cfg) if is_dataclass(cfg) else cfg


def write_run_record(out_dir: str | Path, configs: dict, metrics: dict,
                     labels: np.ndarray | None = None, preds: np.ndarray | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": _git_commit(),
              "python": platform.python_version(), "platform": platform.platform(), "packages": _versions(),
              "configs": {name: _plain(c) for name, c in configs.items()}, "metrics": metrics}
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str))
    if labels is not None and preds is not None:
        (out / "eval.json").write_text(json.dumps({**metrics, "preds": preds.tolist(), "gold": labels.tolist()}))
