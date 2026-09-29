"""Unit tests for the evaluation tools added with the reviewer bake-off, the grounding adapter,
and the real-clip questions. They pin the rules each report depends on."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from scripts_helper import load_script, script_dir

from agentx.services.demo import create_fixture

load = load_script
# The user-study and adapter-gate scripts, run directly below, live with the evaluation scripts.
SCRIPTS = Path(script_dir("eval"))


protocol = load("make_grounding_protocol")
cross_protocol = load("make_cross_reference_protocol")
sft = load("build_grounding_sft")
real = load("evaluate_real_clips")
study = load("prepare_user_study")
bakeoff = load("summarize_vlm_bakeoff")
grounded = load("evaluate_grounded_reviews")
latency = load("measure_live_latency")


def test_live_latency_uses_nearest_rank_for_36_queries():
    # 95% of 36 falls between ranks 34 and 35; nearest rank selects the 35th.
    assert latency.nearest_rank_p95(list(range(1, 37))) == 35
    assert latency.nearest_rank_p95([2.5]) == 2.5


def labels(visible: list[bool], fps: int = 30) -> dict:
    return {
        "assigned_fps": fps,
        "labels": [
            {"at_ms": round(i * 1000 / fps), "visible": v, "box": None}
            for i, v in enumerate(visible)
        ],
    }


def test_window_rule_adds_the_most_absent_window_and_deduplicates():
    frames = [True] * (30 * 40)
    for second in (20, 21, 22):
        frames[second * 30] = False
    chosen = protocol.cutoffs(labels(frames))
    # 7 s, floor(duration / 2), floor(last timestamp), and the earliest window covering all
    # three absent 1 FPS samples (cutoffs 22 .. 27 s cover 20-22 s; 22 is the earliest).
    assert chosen == [7000, 20000, 22000, 39000]
    assert protocol.cutoffs(labels([True] * (30 * 40))) == [7000, 20000, 39000]


def test_window_times_match_the_review_sampler():
    assert protocol.window_times(7000) == [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    assert protocol.window_times(30000)[0] == 23000


def test_cross_reference_protocol_rotates_only_within_category(tmp_path, monkeypatch):
    source = {
        "comparison": "paired prompt transfer",
        "inputs": {
            "book-a": {"category": "book"},
            "cup-a": {"category": "cup"},
            "book-b": {"category": "book"},
            "cup-b": {"category": "cup"},
        },
        "windows": [
            {"sequence": name, "cutoff_ms": 7000, "reference": True}
            for name in ("book-a", "cup-a", "book-b", "cup-b")
        ],
    }
    input_file = tmp_path / "positive.json"
    input_file.write_text(json.dumps(source))
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    output = tmp_path / "cross.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_cross_reference_protocol.py",
            "--source-protocol",
            str(input_file),
            "--selection-plan",
            str(plan),
            "--output",
            str(output),
        ],
    )
    cross_protocol.main()
    result = json.loads(output.read_text())
    assert {row["sequence"]: row["reference_sequence"] for row in result["windows"]} == {
        "book-a": "book-b",
        "book-b": "book-a",
        "cup-a": "cup-b",
        "cup-b": "cup-a",
    }


@pytest.mark.parametrize(
    ("meta", "reason"),
    [
        ({"object_class": "runner", "root_class": "person"}, "root:person"),
        ({"object_class": "face", "root_class": "object part"}, "root:object part"),
        ({"object_class": "beer bottle", "root_class": "passive motion object"}, "bottle"),
        ({"object_class": "african elephant", "root_class": "animal"}, None),
        ({"object_class": "pelican", "root_class": "animal"}, None),
        ({"object_class": "toucan", "root_class": "animal"}, None),
    ],
)
def test_training_data_excludes_people_and_evaluation_categories_by_whole_word(meta, reason):
    assert sft.excluded(meta) == reason


def test_training_labels_follow_got10k_absence_and_cover():
    import numpy as np

    absence = np.array([0, 1, 0, 0])
    cover = np.array([8, 0, 1, 4])
    assert [sft.state(i, absence, cover) for i in range(4)] == [True, False, None, True]
    answer = json.loads(sft.answer(True, (100, 50, 200, 100), 1000, 500, "zebra"))
    assert (answer["x"], answer["y"], answer["present"]) == (200, 200, True)
    absent = json.loads(sft.answer(False, None, 1, 1, "zebra"))
    assert absent["x"] is None and absent["present"] is False


def test_real_clip_grading_needs_status_zone_and_time_window():
    visible = {"status": "visible", "expected_zones": ["center"]}
    assert real.grade(visible, {"status": "visible", "evidence": {"zone": "center"}}, 400)[
        "correct"
    ]
    assert not real.grade(visible, {"status": "visible", "evidence": {"zone": "left"}}, 400)[
        "correct"
    ]
    lost = {"status": "last_seen", "last_seen_window_ms": [3000, 5200], "expected_zones": ["right"]}
    inside = {"status": "last_seen", "evidence": {"zone": "right", "at_ms": 5500}}
    late = {"status": "last_seen", "evidence": {"zone": "right", "at_ms": 5700}}
    assert real.grade(lost, inside, 400)["correct"]
    assert not real.grade(lost, late, 400)["time_ok"]
    assert not real.grade(lost, {"status": "unknown"}, 400)["correct"]
    state = real.object_state({"states": [{"object_id": "a"}, {"object_id": "b", "x": 1}]}, "b")
    assert state == {"object_id": "b", "x": 1}


def test_real_clip_grading_counts_refusals_without_requiring_a_zone():
    for expected in ("unknown", "not_observed"):
        question = {"status": expected}
        assert real.grade(question, {"status": expected}, 400)["correct"]
        assert not real.grade(
            question,
            {"status": "visible", "evidence": {"zone": "center", "at_ms": 8000}},
            400,
        )["correct"]
    with pytest.raises(ValueError, match="Unsupported expected status"):
        real.grade({"status": "moved_to_drawer"}, {}, 400)


def test_real_clip_result_categories_keep_identity_swap_for_frame_review():
    absent = {
        "status": "last_seen",
        "expected_zones": ["right"],
        "last_seen_window_ms": [3000, 5000],
    }
    false_presence = {"status": "visible", "evidence": {"zone": "right", "at_ms": 7000}}
    checks = real.grade(absent, false_presence, 400)
    result = real.classify_result(absent, false_presence, checks, error=None)
    assert result["false_visible"] and result["unsupported_location_claim"]
    assert result["physical_identity_adjudication"] == "not_assessed"
    assert result["frame_review_priority"]
    assert not result["unanswered"]

    visible = {"status": "visible", "expected_zones": ["left"]}
    checks = real.grade(visible, {}, 400)
    result = real.classify_result(visible, {}, checks, error="Index run failed")
    assert result["unanswered"] and not result["unsupported_location_claim"]


def test_real_clip_preflight_checks_all_clips_before_any_http(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    first = clips / "first.mp4"
    second = clips / "second.mp4"
    create_fixture(first)
    shutil.copyfile(first, second)

    def entry(path, split, cutoff):
        return {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "split": split,
            "objects": [{"name": "Toolkit", "at_ms": 1000, "box": [0.1, 0.1, 0.3, 0.4]}],
            "questions": [
                {
                    "object": "Toolkit",
                    "at_ms": cutoff,
                    "status": "visible",
                    "expected_zones": ["left"],
                }
            ],
        }

    spec = {
        "tolerance_ms": 400,
        "clips": [entry(first, "development", 1000), entry(second, "evaluation", 1000)],
    }
    questions = tmp_path / "frozen.json"
    questions.write_text(json.dumps(spec))
    output = tmp_path / "preflight.json"
    monkeypatch.setattr(
        real.httpx, "Client", lambda **_kwargs: pytest.fail("HTTP before preflight")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_real_clips.py",
            "--questions",
            str(questions),
            "--clips",
            str(clips),
            "--output",
            str(output),
            "--preflight-only",
            "--min-evaluation-clips",
            "1",
        ],
    )
    real.main()
    receipt = json.loads(output.read_text())
    assert (receipt["clip_count"], receipt["evaluation_clip_count"], receipt["question_count"]) == (
        2,
        1,
        2,
    )
    assert receipt["questions_sha256"] == hashlib.sha256(questions.read_bytes()).hexdigest()

    spec["clips"][1]["questions"][0]["at_ms"] = 12001
    questions.write_text(json.dumps(spec))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_real_clips.py",
            "--questions",
            str(questions),
            "--clips",
            str(clips),
            "--output",
            str(tmp_path / "invalid.json"),
        ],
    )
    with pytest.raises(ValueError, match="cutoff is outside second.mp4"):
        real.main()
    assert not (tmp_path / "invalid.json").exists()


def study_inputs():
    clips = []
    for number in range(4):
        status = "visible" if number < 2 else "last_seen"
        question = {
            "object": "Toolkit",
            "at_ms": 1000 if number < 2 else 7000,
            "status": status,
            "expected_zones": ["left" if number < 2 else "right"],
        }
        if status == "last_seen":
            question["last_seen_window_ms"] = [5000, 6000]
        clips.append(
            {
                "file": f"clip-{number}.mp4",
                "sha256": str(number),
                "split": "evaluation",
                "questions": [question],
            }
        )
    pairs = {
        "pairs": [
            {
                "id": "visible",
                "match_rationale": "Two matched visible tasks.",
                "tasks": [{"file": f"clip-{i}.mp4", "question_index": 0} for i in (0, 1)],
            },
            {
                "id": "last-seen",
                "match_rationale": "Two matched absence tasks.",
                "tasks": [{"file": f"clip-{i}.mp4", "question_index": 0} for i in (2, 3)],
            },
        ]
    }
    return {"tolerance_ms": 400, "clips": clips}, pairs


def test_user_study_schedule_balances_order_and_clip_without_gold_labels():
    from collections import Counter

    spec, pairs = study_inputs()
    rows = study.build_schedule(spec, pairs, 6)
    assert len(rows) == 24
    assert Counter(row["method"] for row in rows if row["order"] == 1) == {
        "manual": 3,
        "findback": 3,
    }
    assert set(Counter((row["clip_file"], row["method"]) for row in rows).values()) == {3}
    assert all("expected_zones" not in row and "status_for_validation" not in row for row in rows)
    pairs["pairs"][1]["tasks"][0]["file"] = "clip-0.mp4"
    with pytest.raises(ValueError, match="repeated"):
        study.build_schedule(spec, pairs, 6)


def test_user_study_scoring_keeps_failures_and_rejects_missing_attempts(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec, pairs = study_inputs()
    assignments = study.build_schedule(spec, pairs, 4)
    schedule = {"questions_sha256": "frozen", "pairs_sha256": "paired", "assignments": assignments}
    outcomes = study.blank_outcomes(assignments)
    questions = {clip["file"]: clip["questions"][0] for clip in spec["clips"]}
    for row in outcomes:
        gold = questions[row["clip_file"]]
        row.update(
            total_seconds=20 if row["method"] == "manual" else 35,
            setup_seconds=0 if row["method"] == "manual" else 15,
            status=gold["status"],
            zone=gold["expected_zones"][0],
            evidence_at_ms=5500 if gold["status"] == "last_seen" else None,
        )
    manual = next(row for row in outcomes if row["method"] == "manual")
    manual.update(status="no_answer", zone=None, evidence_at_ms=None)
    false_current = next(
        row for row in outcomes if row["method"] == "findback" and row["pair_id"] == "last-seen"
    )
    false_current.update(status="visible", evidence_at_ms=None)
    report = study.score(schedule, outcomes, spec)
    assert report["by_method"]["manual"]["correct"] == 7
    assert report["by_method"]["findback"]["correct"] == 7
    assert report["by_method"]["findback"]["unsupported_location_claims"] == 1
    assert report["by_method"]["manual"]["median_total_seconds"] == 20
    with pytest.raises(ValueError, match="Outcome count"):
        study.score(schedule, outcomes[:-1], spec)


def test_user_study_future_evidence_cannot_pass_grading(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec, pairs = study_inputs()
    assignments = study.build_schedule(spec, pairs, 4)
    schedule = {"questions_sha256": "frozen", "pairs_sha256": "paired", "assignments": assignments}
    outcomes = study.blank_outcomes(assignments)
    for row in outcomes:
        row.update(total_seconds=10, setup_seconds=0, status="no_answer")
    visible = next(row for row in outcomes if row["clip_file"] == "clip-0.mp4")
    visible.update(status="visible", zone="left", evidence_at_ms=1500)
    report = study.score(schedule, outcomes, spec)
    future = next(row for row in report["rows"] if row["clip_file"] == "clip-0.mp4")
    assert future["causal_time_violation"]
    assert not future["correct"]
    assert future["unsupported_location_claim"]


def test_user_study_refusal_cannot_claim_a_location(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec, pairs = study_inputs()
    for clip in spec["clips"][2:]:
        question = clip["questions"][0]
        question.update(status="unknown", expected_zones=[])
        question.pop("last_seen_window_ms")
    assignments = study.build_schedule(spec, pairs, 4)
    schedule = {"questions_sha256": "frozen", "pairs_sha256": "paired", "assignments": assignments}
    outcomes = study.blank_outcomes(assignments)
    for row in outcomes:
        row.update(total_seconds=10, setup_seconds=0, status="no_answer")
    refusal = next(row for row in outcomes if row["clip_file"] == "clip-2.mp4")
    refusal.update(status="unknown", zone="right")
    report = study.score(schedule, outcomes, spec)
    claim = next(row for row in report["rows"] if row["clip_file"] == "clip-2.mp4")
    assert not claim["correct"]
    assert claim["unsupported_location_claim"]


def row(visible: bool, predicted: bool | None, inside: bool, x=0.5, y=0.5) -> dict:
    return {
        "truth_visible": visible,
        "truth_box": None,
        "predicted_visible": predicted,
        "point_inside_target_box": inside,
        "available": predicted is not None,
        "x": x,
        "y": y,
    }


def test_bakeoff_score_and_two_model_panel():
    a = {("s", 1, 0): row(True, True, True), ("s", 1, 1): row(False, True, False)}
    b = {("s", 1, 0): row(True, True, True, x=0.55), ("s", 1, 1): row(False, False, False)}
    single = bakeoff.score(list(a.values()))
    assert (single["correct_point_in_box"], single["false_visible"]) == (1, 1)
    panel = bakeoff.score(bakeoff.consensus(a, b, agree=0.1))
    assert (panel["correct_point_in_box"], panel["false_visible"]) == (1, 0)
    far = {k: {**v, "x": 0.9} for k, v in b.items()}
    assert bakeoff.score(bakeoff.consensus(a, far, agree=0.1))["correct_point_in_box"] == 0
    veto = bakeoff.score(bakeoff.presence_veto(a, far))
    # The veto concerns presence, not point proximity: A's correct point survives even when
    # B's point is far away, while the absent-frame claim is suppressed.
    assert (veto["correct_point_in_box"], veto["false_visible"]) == (1, 0)


def test_cross_recording_negative_preserves_source_labels_and_checks_identity():
    source = {
        "video_sha256": "source",
        "category": "bottle",
        "labels": [{"frame": 1, "at_ms": 0, "visible": True, "box": {"x1": 0.1}, "zone": "left"}],
    }
    reference = {**source, "video_sha256": "reference"}
    absent = grounded.cross_recording_absence(source, reference)
    assert (absent["labels"][0]["visible"], absent["labels"][0]["box"]) == (False, None)
    assert source["labels"][0]["visible"] is True
    with pytest.raises(ValueError, match="different source videos"):
        grounded.cross_recording_absence(source, source)
    with pytest.raises(ValueError, match="same object category"):
        grounded.cross_recording_absence(source, {**reference, "category": "cup"})


def gate_report(path: Path, correct: int, claims: int, false_visible: int, frames=20, absent=2):
    rows = [
        {
            "at_ms": i,
            "truth_visible": True,
            "predicted_visible": i < claims,
            "point_inside_target_box": i < correct,
            "available": True,
        }
        for i in range(frames)
    ] + [
        {
            "at_ms": frames + i,
            "truth_visible": False,
            "predicted_visible": i < false_visible,
            "point_inside_target_box": False,
            "available": True,
        }
        for i in range(absent)
    ]
    window = {"reference": True, "sequence": "s", "cutoff_ms": 9000, "rows": rows}
    path.write_text(json.dumps({"windows": [window]}))
    return str(path)


def run_gate(tmp_path, held: tuple, dev: tuple) -> dict:
    args = [
        "--base-heldout",
        gate_report(tmp_path / "bh.json", *held[0]),
        "--adapter-heldout",
        gate_report(tmp_path / "ah.json", *held[1]),
        "--base-dev",
        gate_report(tmp_path / "bd.json", *dev[0]),
        "--adapter-dev",
        gate_report(tmp_path / "ad.json", *dev[1]),
    ]
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "apply_adapter_gate.py"), *args],
        capture_output=True,
        text=True,
    )
    return json.loads(done.stdout) if done.returncode == 0 else {"error": done.stderr}


def test_adapter_gate_is_exact_at_the_five_point_boundary_and_guards_absences(tmp_path):
    # 10/20 -> 11/20 is exactly +5 points; precision 10/12 -> 11/13 does not fall.
    result = run_gate(tmp_path, ((10, 12, 0), (11, 13, 0)), ((5, 5, 1), (5, 5, 1)))
    assert result["checks"]["heldout_recall_gain_pp"] == 5.0
    assert result["promote"] is True
    assert (
        run_gate(tmp_path, ((10, 12, 0), (11, 13, 0)), ((5, 5, 1), (5, 5, 2)))["promote"] is False
    )  # one more false "visible" on the development absences
    assert (
        run_gate(tmp_path, ((10, 12, 0), (11, 16, 0)), ((5, 5, 0), (5, 5, 0)))["promote"] is False
    )  # recall rose but precision fell (10/12 -> 11/16)
    assert (
        run_gate(tmp_path, ((10, 12, 0), (10, 11, 0)), ((5, 5, 0), (5, 5, 0)))["promote"] is False
    )  # no recall gain


def test_adapter_gate_refuses_reports_that_cover_different_frames(tmp_path):
    base = gate_report(tmp_path / "b.json", 10, 12, 0)
    other = gate_report(tmp_path / "o.json", 10, 12, 0, frames=19)
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "apply_adapter_gate.py"),
            *("--base-heldout", base, "--adapter-heldout", other),
            *("--base-dev", base, "--adapter-dev", base),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0 and "do not cover the same frames" in done.stderr
