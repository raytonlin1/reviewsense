"""Online experiments (A/B tests): does the new model help real users? Offline scores (Parts 2-14) decide whether a
model may be tried; an A/B test decides whether it ships. A random half of users get the new version (treatment),
the rest the current one (control), and a business metric is compared, e.g. "share of chat sessions with a booking".

  1. plan      - how many users are needed to detect the smallest change worth shipping (power analysis)
  2. assign    - each user always gets the same version: a hash of (experiment, user id), not a coin flip per visit
  3. srm_check - did the split come out as designed? A sample ratio mismatch means broken assignment or logging, and
                 then the result can't be trusted, however good it looks
  4. analyze   - difference, relative lift, 95% confidence interval and p-value (two-proportion z-test)
  5. correct   - several metrics tested at once: adjust the p-values (Holm) so one lucky metric isn't a "win"
The rule that matters most: decide the sample size first and analyze once it is reached. Checking every day and
stopping at the first p < 0.05 ("peeking") turns a 5% false-positive rate into far more (simulated in experiments_demo.py).

  python -m experiments.ab_test plan --baseline 0.10 --mde 0.01
  python -m experiments.ab_test analyze --control 10000 1000 --treatment 10000 1100
"""
from __future__ import annotations

import argparse
import hashlib
import json

from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_effectsize, proportions_ztest


def plan(baseline: float, minimum_effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users needed PER GROUP to detect baseline -> baseline + minimum_effect (absolute, e.g. 0.10 -> 0.11) with
    the given power (the chance of seeing a real effect of that size). Smaller effects need many more users."""
    effect_size = proportion_effectsize(baseline + minimum_effect, baseline)
    return int(NormalIndPower().solve_power(effect_size, alpha=alpha, power=power, alternative="two-sided")) + 1


def assign(user_id: str, experiment: str, treatment_share: float = 0.5) -> str:
    """"control" or "treatment", the same every time for this user and experiment. The experiment name is part of
    the hash, so a user's group in one experiment says nothing about their group in the next."""
    digest = hashlib.sha256(f"{experiment}:{user_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10_000                        # 0 .. 9999
    return "treatment" if bucket < treatment_share * 10_000 else "control"


def srm_check(control_users: int, treatment_users: int, treatment_share: float = 0.5, alpha: float = 0.001) -> dict:
    """Chi-square test of the observed split against the designed one. A very small p-value (below 0.001 is the
    usual threshold) means the split is broken: find the bug before reading any result."""
    total = control_users + treatment_users
    expected = [total * (1 - treatment_share), total * treatment_share]
    p_value = float(chisquare([control_users, treatment_users], f_exp=expected).pvalue)
    return {"p_value": round(p_value, 6), "mismatch": p_value < alpha}


def analyze(control: tuple[int, int], treatment: tuple[int, int], alpha: float = 0.05) -> dict:
    """control and treatment: (users, conversions). Returns the effect with its confidence interval: report the
    interval, not only the p-value ("+1.0 point, 95% CI [+0.2, +1.8]" says how big the effect could be)."""
    (control_users, control_conversions), (treatment_users, treatment_conversions) = control, treatment
    control_rate = control_conversions / control_users
    treatment_rate = treatment_conversions / treatment_users
    _, p_value = proportions_ztest([treatment_conversions, control_conversions], [treatment_users, control_users])
    low, high = confint_proportions_2indep(treatment_conversions, treatment_users, control_conversions, control_users,
                                           compare="diff", alpha=alpha)
    return {"control_rate": round(control_rate, 4), "treatment_rate": round(treatment_rate, 4),
            "difference": round(treatment_rate - control_rate, 4),
            "relative_lift": round((treatment_rate - control_rate) / control_rate, 4),
            "ci_95": [round(float(low), 4), round(float(high), 4)], "p_value": round(float(p_value), 4),
            "significant": bool(p_value < alpha)}


def correct(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm's correction for several metrics tested in the same experiment."""
    names = list(p_values)
    reject, adjusted, _, _ = multipletests([p_values[name] for name in names], alpha=alpha, method="holm")
    return {name: {"p_value": p_values[name], "adjusted_p": round(float(p), 4), "significant": bool(r)}
            for name, p, r in zip(names, adjusted, reject)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plan or analyze an A/B test on a conversion rate.")
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="users needed per group")
    planning.add_argument("--baseline", type=float, required=True, help="current rate, e.g. 0.10")
    planning.add_argument("--mde", type=float, required=True, help="smallest change worth detecting, e.g. 0.01")
    planning.add_argument("--power", type=float, default=0.8)
    analysis = commands.add_parser("analyze", help="compare two groups")
    analysis.add_argument("--control", type=int, nargs=2, metavar=("USERS", "CONVERSIONS"), required=True)
    analysis.add_argument("--treatment", type=int, nargs=2, metavar=("USERS", "CONVERSIONS"), required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps({"users_per_group": plan(args.baseline, args.mde, power=args.power)}))
    else:
        result = {"srm": srm_check(args.control[0], args.treatment[0]),
                  "effect": analyze(tuple(args.control), tuple(args.treatment))}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
