from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agentx.domain.contracts import Detection, IndexRequest
from agentx.vision.sam2 import Sam2Detector, mask_box, prune_session


def test_late_registration_is_primed_at_its_source_time_without_future_inputs(monkeypatch):
    detector = Sam2Detector.__new__(Sam2Detector)
    detector.pending = [
        {"id": "a", "registered_at_ms": 350},
        {"id": "b", "registered_at_ms": 400},
        {"id": "future", "registered_at_ms": 800},
    ]
    detector.active = {}
    detector.source = Path("test.mp4")
    detector.last_sample_ms = detector.last_inferred_ms = -1
    detector.last_results = []
    calls, decoded = [], []

    def read(path, at_ms):
        decoded.append(at_ms)
        return at_ms, "registration-frame"

    def advance(frame, at_ms, additions):
        calls.append((frame, at_ms, [o["id"] for o in additions]))
        for obj in additions:
            detector.active[len(detector.active) + 1] = obj
        detector.last_inferred_ms = at_ms
        detector.last_results = [
            Detection(o["id"], None, None, "not_detected") for o in detector.active.values()
        ]

    monkeypatch.setattr("agentx.vision.sam2.read_frame", read)
    detector._advance = advance
    assert detector.detect("first", 0) == []
    assert detector.detect("second", 200) == []
    result = detector.detect("current-frame", 400)
    assert decoded == [350]
    assert calls == [("registration-frame", 350, ["a"]), ("current-frame", 400, ["b"])]
    assert [d.object_id for d in result] == ["a", "b"]
    assert detector.pending == [{"id": "future", "registered_at_ms": 800}]
    with pytest.raises(ValueError, match="increasing"):
        detector.detect("old", 300)


def test_cache_retains_all_registration_frames_and_bounded_recent_history():
    def cache():
        return dict.fromkeys(range(101))

    session = SimpleNamespace(
        processed_frames=cache(),
        frames_tracked_per_obj={0: cache(), 1: cache()},
        output_dict_per_obj={
            0: {"cond_frame_outputs": {0: "reference"}, "non_cond_frame_outputs": cache()},
            1: {"cond_frame_outputs": {50: "late reference"}, "non_cond_frame_outputs": cache()},
        },
    )
    prune_session(session, 100, 32)
    assert set(session.processed_frames) == {0, 50, *range(68, 101)}
    assert session.output_dict_per_obj[1]["cond_frame_outputs"] == {50: "late reference"}
    assert 67 not in session.output_dict_per_obj[0]["non_cond_frame_outputs"]


def test_empty_and_tiny_masks_are_not_visible_evidence():
    mask = np.zeros((1000, 1000), dtype=bool)
    assert mask_box(mask) is None
    mask[50, 80] = True
    assert mask_box(mask) is None
    mask[100:300, 200:400] = True
    # The isolated pixel remains part of the mask's bounds; coordinates are exact.
    box = mask_box(mask)
    assert box is not None and box.x1 == 0.08 and box.y2 == 0.3


def test_missing_sam2_checkpoint_is_rejected_before_enqueue(client, app, indexed):
    video, _ = indexed
    s = app.state.services
    with pytest.raises(ValueError, match="checkpoint"):
        s.indexer.enqueue(video["id"], IndexRequest(backend="sam2"))


@pytest.mark.parametrize(
    ("other_logit", "retired", "ambiguous"),
    [(10.0, False, True), (-10.0, False, False), (10.0, True, False)],
)
def test_overlap_only_uses_present_nonretired_masks(other_logit, retired, ambiguous):
    torch = pytest.importorskip("torch")
    from agentx.vision.presence import PresenceGate

    class Inputs(dict):
        def __getattr__(self, name):
            return self[name]

    detector = Sam2Detector.__new__(Sam2Detector)
    detector.torch = torch
    detector.device, detector.dtype = "cpu", torch.float32
    detector.frame_idx, detector.window = 1, 32
    detector.threshold, detector.appearance = 0.5, None
    detector.gate = PresenceGate(0.5, 0.95, 0.7)
    detector.active = {1: {"id": "target"}, 2: {"id": "companion"}}
    detector.session = SimpleNamespace(
        obj_ids=[1, 2],
        output_dict_per_obj={
            i: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}} for i in range(2)
        },
        processed_frames={},
        frames_tracked_per_obj={},
        get_output=lambda index, *args, **kwargs: torch.tensor(
            [[10.0 if index == 0 else other_logit]]
        ),
    )
    masks = torch.zeros((2, 1, 20, 20))
    masks[:, :, 3:15, 4:16] = 1

    class Processor:
        def __call__(self, **kwargs):
            return Inputs(pixel_values=torch.zeros((1, 3, 20, 20)), original_sizes=[(20, 20)])

        def post_process_masks(self, *args, **kwargs):
            return [masks]

    detector.processor = Processor()
    detector.model = lambda **kwargs: SimpleNamespace(pred_masks=masks)
    detector.collisions = SimpleNamespace(
        retired={"companion"} if retired else set(),
        retire_duplicates=lambda additions, results: None,
        apply=lambda results: results,
    )
    detector._advance(np.zeros((20, 20, 3), dtype=np.uint8), 200, [])
    result = detector.last_results[0]
    assert (result.reason == "identity_ambiguous") == ambiguous
    assert (result.box is None) == ambiguous
