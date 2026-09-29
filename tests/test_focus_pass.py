"""The zoomed second look: geometry, policy and what it is allowed to change."""

import numpy as np
import pytest

from agentx.agents.providers import Providers
from agentx.config import Settings
from agentx.domain.grounding import focus_point, focus_window


def test_a_focus_window_is_square_clamped_and_scaled_to_the_shorter_edge():
    assert focus_window(0.5, 0.5, 1280, 720, 0.5) == (460, 180, 820, 540)
    # A point near a corner keeps the full window by sliding it inside the frame.
    left, top, right, bottom = focus_window(0.01, 0.99, 1280, 720, 0.5)
    assert (right - left, bottom - top) == (360, 360)
    assert (left, top) == (0, 360)


def test_a_point_measured_in_the_crop_maps_back_to_the_frame():
    window = focus_window(0.5, 0.5, 1280, 720, 0.5)
    assert focus_point(0.5, 0.5, window, 1280, 720) == pytest.approx((0.5, 0.5), abs=0.001)
    # The crop's own top-left corner is the window's top-left corner in the frame.
    assert focus_point(0.0, 0.0, window, 1280, 720) == pytest.approx((460 / 1280, 0.25), abs=0.001)


def scripted(monkeypatch, provider, coarse, refined, frame_count=2):
    evidence = [{"id": i, "at_ms": i * 1000} for i in range(frame_count)]
    originals = [np.full((720, 1280, 3), 120, np.uint8) for _ in range(frame_count)]
    monkeypatch.setattr(
        provider,
        "_sample_frames",
        lambda *a, **kw: (evidence, [b"frame"] * frame_count, originals),
    )
    monkeypatch.setattr(provider, "_provenance", lambda: {"backend": "http"})
    answers = []

    def generate(messages, **kwargs):
        # The close-up request is the one that names a crop; a retry keeps that turn in place.
        close_up = "close-up crop" in str(messages)
        answers.append("focus" if close_up else "frame")
        return refined if close_up else coarse

    monkeypatch.setattr(provider, "_generate", generate)
    return evidence, answers


def provider_for(policy: str) -> Providers:
    return Providers(
        Settings(
            cosmos_backend="http",
            cosmos_review_workers=1,
            cosmos_focus=policy,
            cosmos_focus_window=0.5,
            _env_file=None,
        )
    )


def test_the_focus_pass_refines_the_point_it_was_given(monkeypatch):
    provider = provider_for("refine")
    _, answers = scripted(
        monkeypatch,
        provider,
        '{"present":true,"x":0.5,"y":0.5,"note":"coarse"}',
        '{"present":true,"x":0.25,"y":0.75,"note":"close"}',
    )
    result = provider.review("unused", 1000, "Where?", target="cup")
    report = result["frame_reports"][0]
    assert answers == ["frame", "focus", "frame", "focus"]
    # 0.25 inside a 360px window starting at x=460 is 550/1280 of the frame.
    assert report["x"] == pytest.approx(550 / 1280, abs=0.002)
    assert report["y"] == pytest.approx(450 / 720, abs=0.002)
    assert report["focus"]["status"] == "confirmed"
    assert result["provenance"]["focus_pass"] == "refine"
    assert result["provenance"]["focus_confirmed"] == 2


def test_refine_keeps_the_first_answer_when_the_close_up_disagrees(monkeypatch):
    provider = provider_for("refine")
    scripted(
        monkeypatch,
        provider,
        '{"present":true,"x":0.5,"y":0.5,"note":"coarse"}',
        '{"present":false,"x":null,"y":null,"note":"cannot see it"}',
    )
    report = provider.review("unused", 1000, "Where?", target="cup")["frame_reports"][0]
    assert report["present"] and report["x"] == 0.5
    assert report["focus"]["status"] == "unconfirmed"


def test_confirm_reports_uncertainty_when_the_two_looks_disagree(monkeypatch):
    provider = provider_for("confirm")
    scripted(
        monkeypatch,
        provider,
        '{"present":true,"x":0.5,"y":0.5,"note":"coarse"}',
        '{"present":false,"x":null,"y":null,"note":"cannot see it"}',
    )
    result = provider.review("unused", 1000, "Where?", target="cup")
    report = result["frame_reports"][0]
    assert not report["present"] and report["x"] is None
    assert result["provenance"]["focus_unconfirmed"] == 2
    assert "not visible in any of the 2 supplied frames" in result["summary"]


def test_an_absent_first_answer_costs_no_second_request(monkeypatch):
    provider = provider_for("confirm")
    _, answers = scripted(
        monkeypatch,
        provider,
        '{"present":false,"x":null,"y":null,"note":"gone"}',
        '{"present":true,"x":0.5,"y":0.5,"note":"unused"}',
    )
    provider.review("unused", 1000, "Where?", target="cup")
    assert answers == ["frame", "frame"]


def test_a_malformed_close_up_never_overrides_the_first_answer(monkeypatch):
    provider = provider_for("confirm")
    scripted(
        monkeypatch,
        provider,
        '{"present":true,"x":0.5,"y":0.5,"note":"coarse"}',
        "not json at all",
    )
    report = provider.review("unused", 1000, "Where?", target="cup")["frame_reports"][0]
    assert report["present"] and report["x"] == 0.5
    assert report["focus"]["status"] == "unavailable"


def test_the_focus_pass_is_off_by_default(monkeypatch):
    provider = Providers(Settings(cosmos_backend="http", cosmos_review_workers=1, _env_file=None))
    _, answers = scripted(
        monkeypatch,
        provider,
        '{"present":true,"x":0.5,"y":0.5,"note":"coarse"}',
        '{"present":true,"x":0.25,"y":0.75,"note":"close"}',
    )
    result = provider.review("unused", 1000, "Where?", target="cup")
    assert answers == ["frame", "frame"]
    assert result["provenance"]["focus_pass"] == "off"


def scripted_poses(monkeypatch, provider, points, frames=3):
    evidence = [{"id": i, "at_ms": i * 1000} for i in range(frames)]
    monkeypatch.setattr(
        provider,
        "_sample_frames",
        lambda *a, **kw: (evidence, [b"frame"] * frames, [None] * frames),
    )
    monkeypatch.setattr(provider, "_provenance", lambda: {"backend": "http"})
    supply = iter(points)
    monkeypatch.setattr(provider, "_generate", lambda *a, **k: next(supply))
    return evidence


def test_a_review_reads_its_points_in_registration_coordinates(monkeypatch):
    """A camera move must not make the review rename the place, or invent a movement."""
    from agentx.domain.contracts import default_regions

    # The recorded pose for the second frame: the camera panned a quarter of the frame right.
    panned = [[1.0, 0.0, 0.25], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    poses = [
        {"at_ms": 0, "scene_reference": "registered", "scene_transform": None},
        {"at_ms": 1000, "scene_reference": "compensated", "scene_transform": panned},
        {"at_ms": 2000, "scene_reference": "unavailable", "scene_transform": None},
    ]
    # The object rests in the centre third; after the pan the same pixels read as the right third.
    answers = [
        '{"present":true,"x":0.45,"y":0.5,"note":"a"}',
        '{"present":true,"x":0.70,"y":0.5,"note":"b"}',
        '{"present":true,"x":0.70,"y":0.5,"note":"c"}',
    ]

    def run(camera_poses):
        provider = Providers(
            Settings(cosmos_backend="http", cosmos_review_workers=1, _env_file=None)
        )
        scripted_poses(monkeypatch, provider, list(answers))
        return provider.review(
            "unused",
            2000,
            "Where?",
            target="cup",
            regions=default_regions(),
            camera_poses=camera_poses,
        )

    naive = run([])
    assert [r["zone"] for r in naive["frame_reports"]] == ["center", "right", "right"]
    assert "Zones over time" in naive["summary"]  # a movement that never happened

    corrected = run(poses)
    reports = corrected["frame_reports"]
    assert [r["zone"] for r in reports] == ["center", "center", None]
    assert [r["camera"] for r in reports] == [None, "compensated", "unavailable"]
    # The published point always belongs to the frame the model saw.
    assert [r["x"] for r in reports] == [0.45, 0.70, 0.70]
    assert corrected["provenance"]["camera_compensated_frames"] == 1
    assert corrected["provenance"]["camera_unavailable_frames"] == 1
    assert corrected["provenance"]["camera_source"] == "recorded index-run poses"
    assert "Zones over time" not in corrected["summary"]
    # The last visible frame is the one the camera move left unnamed, and the summary says so
    # instead of borrowing a region name for a coordinate frame it does not describe.
    assert "last visible at a position the camera move leaves unnamed" in corrected["summary"]
    assert "could not be related to the registered view" in corrected["summary"]
    assert "Center area" not in corrected["summary"]


def test_a_pose_that_is_not_close_enough_in_time_is_not_used():
    from agentx.domain.grounding import camera_at

    poses = [{"at_ms": 1000, "scene_reference": "compensated", "scene_transform": [[1.0]]}]
    assert camera_at(1100, poses, 200) is poses[0]
    assert camera_at(1400, poses, 200) is None
    assert camera_at(1000, [], 200) is None
