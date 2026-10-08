"""A/B testing functions: planning, assignment, split check, analysis and multiple-testing correction."""
import pytest

from experiments.ab_test import analyze, assign, correct, plan, srm_check


def test_planning():
    assert plan(0.10, 0.01) == 14_745                                  # users per group for 10% -> 11%
    assert plan(0.10, 0.005) > 3.5 * plan(0.10, 0.01)                   # half the effect: ~4x the users
    assert plan(0.10, 0.01, power=0.9) > plan(0.10, 0.01)


def test_assignment_is_stable_balanced_and_independent_between_experiments():
    assert assign("user42", "exp-a") == assign("user42", "exp-a")
    groups = [assign(f"user{i}", "exp-a") for i in range(20_000)]
    assert not srm_check(groups.count("control"), groups.count("treatment"))["mismatch"]
    same = sum(assign(f"user{i}", "exp-a") == assign(f"user{i}", "exp-b") for i in range(20_000)) / 20_000
    assert 0.47 < same < 0.53                                           # no "always treatment" users
    shares = [assign(f"user{i}", "exp-c", treatment_share=0.1) for i in range(20_000)]
    assert 0.09 < shares.count("treatment") / 20_000 < 0.11


def test_sample_ratio_mismatch_is_detected():
    assert srm_check(52_000, 48_000)["mismatch"]
    assert not srm_check(50_100, 49_900)["mismatch"]
    assert not srm_check(90_000, 10_000, treatment_share=0.1)["mismatch"]


def test_analysis():
    result = analyze((10_000, 1_000), (10_000, 1_100))
    assert result["difference"] == pytest.approx(0.01) and result["relative_lift"] == pytest.approx(0.10)
    assert result["ci_95"][0] > 0 and result["significant"] and result["p_value"] == pytest.approx(0.0211, abs=1e-4)
    assert not analyze((1_000, 100), (1_000, 110))["significant"]      # same rates, 10x fewer users


def test_multiple_metrics_are_corrected():
    result = correct({"bookings": 0.01, "thumbs_up": 0.04, "latency": 0.03})
    assert result["bookings"]["significant"]
    assert not result["thumbs_up"]["significant"] and result["thumbs_up"]["adjusted_p"] == pytest.approx(0.06)
