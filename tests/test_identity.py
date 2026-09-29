import numpy as np
import pytest

from agentx.domain.contracts import Detection
from agentx.domain.temporal import EventReducer
from agentx.vision.identity import AppearanceMatcher, crop


def matcher(threshold=0.6, margin=0.05):
    m = AppearanceMatcher.__new__(AppearanceMatcher)
    m.threshold, m.margin = threshold, margin
    return m


def unit(*values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_assignment_confirms_distinct_objects_and_flags_lookalikes():
    a, b = unit(1, 0, 0), unit(0, 1, 0)
    candidates = np.stack([unit(0.9, 0.1, 0), unit(0.1, 0.9, 0), unit(0, 0, 1)])
    result = matcher().assign({"a": a, "b": b}, candidates)
    assert result["a"][0] == 0 and result["a"][2] is None
    assert result["b"][0] == 1 and result["b"][2] is None
    # Two near-identical candidates within the margin are ambiguous, never silently chosen.
    twins = np.stack([unit(0.9, 0.1, 0), unit(0.9, 0.12, 0)])
    result = matcher().assign({"a": a}, twins)
    assert result["a"][0] is None and result["a"][2] == "identity_ambiguous"
    # A candidate below the threshold is not the registered object.
    weak = matcher().assign({"a": a}, np.stack([unit(0.3, 0.3, 0.9)]))
    assert weak["a"][0] is None and weak["a"][2] == "identity_unconfirmed"
    assert weak["a"][1] == pytest.approx(float(unit(0.3, 0.3, 0.9) @ a), abs=1e-6)
    # Two registered objects that match one candidate equally well are both ambiguous.
    shared = matcher().assign({"a": a, "c": unit(0.95, 0.05, 0)}, np.stack([unit(1, 0, 0)]))
    assert {v[2] for v in shared.values()} == {"identity_ambiguous"}
    # With a clear winner the candidate goes to the closest object only.
    clear = matcher().assign({"a": a, "c": unit(0.8, 0.6, 0)}, np.stack([unit(1, 0, 0)]))
    assert clear["a"][0] == 0 and clear["c"][0] is None
    assert matcher().assign({"a": a}, np.zeros((0, 3), dtype=np.float32))["a"][2] == "not_detected"


def test_identity_reasons_flow_into_causal_events():
    reducer = EventReducer([])
    assert (
        reducer.update(0, [Detection("o", None, None, "identity_unconfirmed", identity=0.4)])[
            0
        ].kind
        == "ambiguous"
    )
    events = reducer.update(200, [Detection("o", None, None, "identity_ambiguous", identity=0.7)])
    assert events[0].kind == "ambiguous" and events[0].reason == "identity_ambiguous"


def test_crop_is_padded_and_clipped():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert crop(frame, (0, 0, 50, 50)).shape[:2] == (55, 55)
    assert crop(frame, (150, 60, 200, 100)).shape[:2] == (44, 54)
