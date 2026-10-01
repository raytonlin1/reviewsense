"""Release gate: is the candidate model really better than the baseline/current model on the same test set?

  * bootstrap CI (scipy.stats.bootstrap, paired)   - 95% interval for the accuracy difference (candidate - baseline)
  * McNemar's exact test (statsmodels)             - paired classifiers; uses only the items they disagree on
  * paired permutation test (scipy.stats)          - any per-item metric (F1, ROUGE, WER, nDCG)

  python -m experiments.significance artifacts/tfidf-baseline artifacts/distilbert-20k
  python -m experiments.significance artifacts/tfidf-baseline artifacts/distilbert-20k --gate   # exit 1 unless better
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, permutation_test
from statsmodels.stats.contingency_tables import mcnemar


def compare(correct_baseline, correct_candidate, seed: int = 0) -> dict:
    a, b = np.asarray(correct_baseline, float), np.asarray(correct_candidate, float)
    diff = lambda x, y: y.mean() - x.mean()
    ci = bootstrap((a, b), diff, paired=True, vectorized=False, n_resamples=10_000, confidence_level=0.95,
                   random_state=seed).confidence_interval
    table = [[int(((a == 1) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())],
             [int(((a == 0) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())]]
    perm = permutation_test((a, b), diff, permutation_type="samples", n_resamples=10_000, random_state=seed)
    return {"baseline_acc": round(float(a.mean()), 4), "candidate_acc": round(float(b.mean()), 4),
            "diff": round(float(b.mean() - a.mean()), 4), "diff_95ci": [round(float(ci.low), 4), round(float(ci.high), 4)],
            "mcnemar_p": round(float(mcnemar(table, exact=True).pvalue), 4), "permutation_p": round(float(perm.pvalue), 4)}


def passes_gate(result: dict, alpha: float = 0.05) -> bool:
    """Ship only if the candidate is better AND the whole confidence interval is above zero AND p < alpha."""
    return result["diff"] > 0 and result["diff_95ci"][0] > 0 and result["mcnemar_p"] < alpha


def load_eval(run: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(run)
    data = json.loads((path / "eval.json" if path.is_dir() else path).read_text())
    return np.array(data["preds"]), np.array(data["gold"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline", help="model dir (or eval.json) of the current / baseline model")
    ap.add_argument("candidate", help="model dir (or eval.json) of the new model")
    ap.add_argument("--gate", action="store_true", help="exit code 1 unless the candidate is significantly better")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)
    (base_pred, base_gold), (cand_pred, cand_gold) = load_eval(args.baseline), load_eval(args.candidate)
    if len(base_gold) != len(cand_gold) or not (base_gold == cand_gold).all():   # same reviews, same order
        raise SystemExit("runs were scored on different test sets; the comparison is meaningless")
    base, cand = base_pred == base_gold, cand_pred == cand_gold
    result = compare(base, cand) | {"passes_gate": None}
    result["passes_gate"] = passes_gate(result, args.alpha)
    print(json.dumps(result))
    return 0 if (result["passes_gate"] or not args.gate) else 1


if __name__ == "__main__":
    sys.exit(main())
