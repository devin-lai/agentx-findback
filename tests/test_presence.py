import numpy as np

from agentx.vision.presence import PresenceGate


def test_occluder_cannot_teach_the_guard_its_own_appearance():
    gate = PresenceGate(minimum=0.5, confident=0.95, similarity=0.7)
    target, occluder = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    gate.seed(1, target)
    for _ in range(20):
        assert gate.admit(1, 0.92, occluder) == (False, 0.0)
    assert len(gate.banks[1]) == 1
    assert gate.admit(1, 0.8, target) == (True, 1.0)
    assert gate.admit(1, 0.4, target) == (False, 1.0)


def test_presence_guard_keeps_reference_and_bounded_views_separate_per_object():
    gate = PresenceGate(minimum=0.5, confident=0.95, similarity=0.7)
    original = np.array([1.0, 0.0])
    later = np.array([0.0, 1.0])
    gate.seed(1, original)
    gate.seed(2, later)
    for _ in range(20):
        assert gate.admit(1, 0.99, later)[0]
    assert len(gate.banks[1]) == 9 and len(gate.banks[2]) == 1
    assert np.array_equal(gate.banks[1][0], original)
    assert gate.admit(2, 0.8, original) == (False, 0.0)
    # An installation without appearance weights uses the stricter presence score.
    assert gate.admit(3, 0.92, None) == (False, None)
    assert gate.admit(3, 0.99, None) == (True, None)
