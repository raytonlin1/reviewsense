"""Release gate: is the candidate model really better than the baseline/current model on the same test set?

  * McNemar's exact test (statsmodels): the standard test for two classifiers scored on the same items. It uses only
    the reviews where they disagree (one right, the other wrong).
  * Bootstrap confidence interval (scipy): how big the improvement is, e.g. +3.7 points, 95% CI [+1.3, +6.0].

  python -m experiments.significance artifacts/tfidf-baseline artifacts/distilbert-20k
  python -m experiments.significance artifacts/tfidf-baseline artifacts/distilbert-20k --gate   # exit 1 unless better
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from statsmodels.stats.contingency_tables import mcnemar


def accuracy_difference(baseline_correct, candidate_correct):
    return candidate_correct.mean() - baseline_correct.mean()


def compare(baseline_correct, candidate_correct, seed: int = 0) -> dict:
    """Inputs: one True/False per test review (was each model right?)."""
    a, b = np.asarray(baseline_correct, bool), np.asarray(candidate_correct, bool)
    ci = bootstrap((a, b), accuracy_difference, paired=True, vectorized=False, n_resamples=10_000,
                   random_state=seed).confidence_interval
    # 2x2 table: [both right, only baseline right], [only candidate right, both wrong]
    table = [[(a & b).sum(), (a & ~b).sum()],
             [(~a & b).sum(), (~a & ~b).sum()]]
    return {"baseline_acc": round(float(a.mean()), 4),
            "candidate_acc": round(float(b.mean()), 4),
            "diff": round(float(b.mean() - a.mean()), 4),
            "diff_95ci": [round(float(ci.low), 4), round(float(ci.high), 4)],
            "mcnemar_p": round(float(mcnemar(table, exact=True).pvalue), 4)}


def passes_gate(result: dict, alpha: float = 0.05) -> bool:
    """Ship only if the candidate is better AND the whole confidence interval is above zero AND p < alpha."""
    return result["diff"] > 0 and result["diff_95ci"][0] > 0 and result["mcnemar_p"] < alpha


def load_eval(run: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads((Path(run) / "eval.json").read_text())
    return np.array(data["preds"]), np.array(data["gold"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two models scored on the same test set.")
    parser.add_argument("baseline", help="model dir of the current / baseline model")
    parser.add_argument("candidate", help="model dir of the new model")
    parser.add_argument("--gate", action="store_true", help="exit code 1 unless the candidate is significantly better")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    base_pred, base_gold = load_eval(args.baseline)
    cand_pred, cand_gold = load_eval(args.candidate)
    if not np.array_equal(base_gold, cand_gold):                 # same reviews, same order, same labels
        raise SystemExit("runs were scored on different test sets; the comparison is meaningless")
    result = compare(base_pred == base_gold, cand_pred == cand_gold)
    result["passes_gate"] = passes_gate(result, args.alpha)
    print(json.dumps(result))
    if args.gate and not result["passes_gate"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
