"""Part 15 demo: planning and analyzing an A/B test, and why you must not peek. Outcomes here are SIMULATED (no real
users yet); the statistics are the ones a production experiment would use.

    python experiments_demo.py
"""
import numpy as np

from experiments.ab_test import analyze, assign, correct, plan, srm_check

rng = np.random.default_rng(7)

# 1. Planning: how many users per group to detect a change in a 10% booking rate?
print("Users needed per group (baseline booking rate 10%, 80% power, 5% false-positive rate):")
for points in [0.005, 0.01, 0.02]:
    print(f"  to detect +{points * 100:.1f} points: {plan(0.10, points):,}")
print("  Half the effect needs about four times the users.\n")

# 2. Assignment: hash of (experiment, user id), so a returning user always sees the same version.
users = [f"user{i}" for i in range(30_000)]
groups = [assign(user, "chatbot-sft-0.6b") for user in users]
control_users, treatment_users = groups.count("control"), groups.count("treatment")
print(f"Assigned: {control_users:,} control, {treatment_users:,} treatment  ->  SRM check {srm_check(control_users, treatment_users)}")

# 3. A simulated experiment: the treatment truly raises bookings from 10.0% to 11.0%.
bookings_control = int(rng.binomial(control_users, 0.10))
bookings_treatment = int(rng.binomial(treatment_users, 0.11))
result = analyze((control_users, bookings_control), (treatment_users, bookings_treatment))
print(f"Booking rate {result['control_rate']:.2%} -> {result['treatment_rate']:.2%}: "
      f"{result['difference'] * 100:+.2f} points, 95% CI [{result['ci_95'][0] * 100:+.2f}, "
      f"{result['ci_95'][1] * 100:+.2f}], p = {result['p_value']}")
print("  (The true effect is +1.0 point and the groups are about the planned size, yet it was found only just. With\n"
      "   80% power, 1 in 5 such experiments misses it. The interval shows how uncertain the result is.)\n")

# 4. Several metrics in one experiment: correct for multiple tests (Holm).
metrics = {"booking rate": result["p_value"], "answer thumbs-up": 0.04, "safety blocks": 0.30, "session length": 0.02}
print("Several metrics (the last three p-values are made up for the example):")
for name, scores in correct(metrics).items():
    print(f"  {name:17} p = {scores['p_value']:<7} adjusted p = {scores['adjusted_p']:<7} "
          f"{'significant' if scores['significant'] else 'not significant'}")
print()

# 5. Peeking: 2,000 A/A tests (both groups get the SAME product, so every "win" is false), 14 days, 1,000 users per
#    group per day. Analyzing once at day 14 vs. checking every day and stopping at the first p < 0.05.
days, per_day, simulations = 14, 1000, 2000
false_once = false_peeking = 0
for _ in range(simulations):
    a = rng.binomial(per_day, 0.10, size=days).cumsum()
    b = rng.binomial(per_day, 0.10, size=days).cumsum()
    users_so_far = per_day * np.arange(1, days + 1)
    p_values = [analyze((int(n), int(x)), (int(n), int(y)))["p_value"] for n, x, y in zip(users_so_far, a, b)]
    false_once += p_values[-1] < 0.05
    false_peeking += min(p_values) < 0.05
print(f"False 'wins' in {simulations:,} tests with no real difference:")
print(f"  analyzed once, at the planned end: {false_once / simulations:.1%}   (designed: 5%)")
print(f"  checked daily, stop at p < 0.05:   {false_peeking / simulations:.1%}")
