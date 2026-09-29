"""The ablation's paired statistic and scoring, which decide a published claim."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))

from evaluate_skill_ablation import mcnemar_exact, score  # noqa: E402
from summarize_skill_ablation import mcnemar as summarize_mcnemar  # noqa: E402


def test_mcnemar_ignores_agreement_and_reports_no_effect():
    """Concordant pairs carry no information about a difference, however many there are."""
    result = mcnemar_exact(0, 0)
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        # Two-sided exact binomial on the discordant pairs: 2 * P(X <= min(b, c)), X ~ Bin(b+c, ½).
        (1, 0, 1.0),  # 2 * (1/2)
        (2, 0, 0.5),  # 2 * (1/4)
        (5, 0, 0.0625),  # 2 * (1/32)
        (10, 0, 0.001953),
        (0, 10, 0.001953),  # symmetric: direction is read from b and c, not from p
        (8, 2, 0.109375),
        (6, 6, 1.0),  # capped at 1
    ],
)
def test_mcnemar_exact_matches_the_binomial_tail(b, c, expected):
    assert mcnemar_exact(b, c)["p_value"] == pytest.approx(expected, abs=5e-7)


def test_mcnemar_direction_is_recoverable():
    """b and c must not be collapsed: a gain and a regression share the same p-value."""
    gain, regression = mcnemar_exact(7, 1), mcnemar_exact(1, 7)
    assert gain["p_value"] == regression["p_value"]
    assert (gain["b"], gain["c"]) == (7, 1)
    assert (regression["b"], regression["c"]) == (1, 7)


def row(number, arm, passed):
    return {
        "suite": "s",
        "number": number,
        "arm": arm,
        "passed": passed,
        "checks": {"object": passed},
        "case": {"text": f"case {number}"},
    }


def test_summary_pairs_arms_by_case_not_by_position():
    rows = [
        row(0, "agent_no_skills", False),
        row(1, "agent_no_skills", True),
        row(2, "agent_no_skills", True),
        # Deliberately out of order: pairing must use (suite, number).
        row(2, "agent_skills", False),
        row(0, "agent_skills", True),
        row(1, "agent_skills", True),
    ]
    result = summarize_mcnemar(rows)
    assert (result["b"], result["c"]) == (1, 1)
    assert {f["number"] for f in result["flips"]} == {0, 2}


def test_both_implementations_agree():
    rows = [row(n, "agent_no_skills", n % 3 == 0) for n in range(12)]
    rows += [row(n, "agent_skills", n % 2 == 0) for n in range(12)]
    summarized = summarize_mcnemar(rows)
    direct = mcnemar_exact(summarized["b"], summarized["c"])
    assert direct["p_value"] == summarized["p_value"]


class FakeMemory:
    def __init__(self, owner):
        self.owner = owner

    def observation(self, observation_id):
        return (type("Obs", (), {"object_id": self.owner})(),)


def services_for(owner):
    return type("S", (), {"memory": FakeMemory(owner)})()


def answer(**overrides):
    base = {
        "states": [{"name": "Red toolkit", "status": "visible", "zone": "left"}],
        "intent": "location",
        "evidence": [{"at_ms": 1000, "video_id": "v", "run_id": "r", "observation_id": "o"}],
        "answer": "text",
    }
    return {**base, **overrides}


def case(**overrides):
    base = {
        "text": "Where is Red toolkit?",
        "at_ms": 2000,
        "expected_object": "Red toolkit",
        "expected_intent": "location",
    }
    return {**base, **overrides}


def test_score_rejects_evidence_after_the_cutoff():
    checks = score(
        case(at_ms=500),
        answer(),
        services_for("red"),
        "v",
        "r",
        {"Red toolkit": "red"},
    )
    assert checks["causal_evidence"] is False


def test_score_rejects_evidence_belonging_to_another_object():
    checks = score(case(), answer(), services_for("blue"), "v", "r", {"Red toolkit": "red"})
    assert checks["object_scope"] is False


def test_score_requires_silence_when_no_object_is_expected():
    expected_none = case(expected_object=None)
    claimed = score(expected_none, answer(), services_for("red"), "v", "r", {})
    assert claimed["no_unsupported_claim"] is False
    refused = score(
        expected_none,
        answer(states=[], evidence=[]),
        services_for("red"),
        "v",
        "r",
        {},
    )
    assert refused["no_unsupported_claim"] is True
    assert all(refused.values())


def test_the_held_out_suite_stays_frozen():
    """Its labels back a published claim; an edit must be a deliberate, visible change."""
    import hashlib

    path = Path(__file__).resolve().parents[1] / "tests/fixtures/planner_skill_holdout_v1.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "50148ace67637332c36a4c925e702bdb640ddb5621ae8593924d6356ce494167"
    dataset = json.loads(path.read_text())
    assert len(dataset["cases"]) == 28
    # Every case carries the fields the harness scores against.
    for entry in dataset["cases"]:
        assert {"text", "at_ms", "expected_object", "expected_intent"} <= set(entry)
