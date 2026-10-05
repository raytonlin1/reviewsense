"""Fast safety tests: personal data, claim cleaning, guard decisions and the test sets (no model downloads:
Presidio uses the local spaCy model; the guard's models are replaced by stand-ins)."""
import yaml

from reviewsense.config import get_settings
from reviewsense.safety.fact_check import answer_sentences, claim_text, premise
from reviewsense.safety.input_guard import InputGuard
from reviewsense.safety.pii import find_pii, redact


def test_personal_data_is_removed():
    text = "Call me at 206-555-0147 or email jane.doe@gmail.com. They charged my card 4111 1111 1111 1111 twice."
    assert redact(text) == "Call me at <PHONE_NUMBER> or email <EMAIL_ADDRESS>. They charged my card <CREDIT_CARD> twice."
    assert find_pii(text) == ["CREDIT_CARD", "EMAIL_ADDRESS", "PHONE_NUMBER"]


def test_useful_numbers_and_staff_names_are_kept():
    for text in ["Prices are fair at about $12 per bowl.", "Located at 47.6097, -122.3331 downtown.",
                 "Our server Maria was attentive.", "We waited 45 minutes for table 12."]:
        assert redact(text) == text


def test_claims_lose_citations_and_attributions():
    assert claim_text("Review [3] mentions that Bella Napoli is overpriced.") == "Bella Napoli is overpriced."
    assert claim_text("The salsa verde is fresh [1][2].") == "The salsa verde is fresh."
    assert premise("Never again.", "Bella Napoli Pizzeria") == "A review of Bella Napoli Pizzeria says: Never again."


def test_answer_sentences_keep_their_citations():
    assert answer_sentences("Review [3] says it is overpriced. Review [4] says it is cheap.") == [
        "Review [3] says it is overpriced.", "Review [4] says it is cheap."]
    assert answer_sentences("They waited 45 minutes. [1] No other reviews mention it.") == [
        "They waited 45 minutes. [1]", "No other reviews mention it."]


class StandInGuard(InputGuard):
    """The guard's decision logic with fixed model outputs."""
    def __init__(self, injection: bool, label: str):
        self.injection_result, self.label = injection, label

    def is_injection(self, text):
        return self.injection_result

    def classify(self, text):
        return self.label


def test_guard_decisions():
    assert StandInGuard(True, "ALLOWED").check("x") == "injection"            # injection is checked first
    assert StandInGuard(False, "ALLOWED").check("x") is None
    assert StandInGuard(False, "HARMFUL.").check("x") == "harmful"
    assert StandInGuard(False, "OFF_TOPIC").check("x") == "off_topic"
    assert StandInGuard(False, "I think it's fine").check("x") == "unclear"   # fail closed


def test_red_team_sets_have_attacks_and_normal_questions():
    for name in ["red_team.yaml", "red_team_holdout.yaml"]:
        rows = yaml.safe_load((get_settings().data_dir / name).read_text())
        kinds = {row["kind"] for row in rows}
        assert kinds == {"injection", "off_topic", "harmful", "normal"}
        assert all((row["expect"] == "answer") == (row["kind"] == "normal") for row in rows)
