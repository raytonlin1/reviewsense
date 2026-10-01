"""Is model B really better than model A on the same test set?

  * bootstrap CI (scipy.stats.bootstrap, paired)   - 95% interval for the accuracy difference
  * McNemar's exact test (statsmodels)             - paired classifiers; uses only the items they disagree on
  * paired permutation test (scipy.stats)          - any per-item metric (F1, ROUGE, WER, nDCG)

  python -m experiments.significance artifacts/bert-yelp-eval.json artifacts/distilbert-yelp-eval.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import bootstrap, permutation_test
from statsmodels.stats.contingency_tables import mcnemar


def compare(correct_a, correct_b, seed: int = 0) -> dict:
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    ci = bootstrap((a, b), lambda x, y: x.mean() - y.mean(), paired=True, vectorized=False,
                   n_resamples=10_000, confidence_level=0.95, random_state=seed).confidence_interval
    table = [[int(((a == 1) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())],
             [int(((a == 0) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())]]
    perm = permutation_test((a, b), lambda x, y: x.mean() - y.mean(), permutation_type="samples",
                            n_resamples=10_000, random_state=seed)
    return {"acc_a": round(float(a.mean()), 4), "acc_b": round(float(b.mean()), 4), "diff": round(float(a.mean() - b.mean()), 4),
            "diff_95ci": (round(float(ci.low), 4), round(float(ci.high), 4)),
            "mcnemar_p": round(float(mcnemar(table, exact=True).pvalue), 4), "permutation_p": round(float(perm.pvalue), 4)}


if __name__ == "__main__":
    A, B = (json.load(open(p)) for p in sys.argv[1:3])
    print(compare(np.array(A["preds"]) == np.array(A["gold"]), np.array(B["preds"]) == np.array(B["gold"])))
