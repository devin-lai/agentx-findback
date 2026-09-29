from threading import Barrier, Lock

import pytest

from agentx.agents.providers import Providers
from agentx.config import Settings


@pytest.mark.parametrize(
    "backend,workers,frame_count,expected_parallelism",
    [
        ("http", 1, 8, 1),
        ("http", 2, 8, 2),
        ("http", 4, 8, 4),
        ("http", 8, 8, 8),
        ("http", 8, 2, 2),
        ("transformers", 8, 4, 1),
    ],
)
def test_review_parallelism_is_bounded_and_preserves_frame_order(
    monkeypatch, backend, workers, frame_count, expected_parallelism
):
    provider = Providers(
        Settings(cosmos_backend=backend, cosmos_review_workers=workers, _env_file=None)
    )
    evidence = [{"id": i, "at_ms": i * 1000} for i in range(frame_count)]
    monkeypatch.setattr(
        provider,
        "_sample_frames",
        lambda *a, **kw: (evidence, [b"frame"] * frame_count, [None] * frame_count),
    )
    monkeypatch.setattr(provider, "_provenance", lambda: {"backend": backend})
    barrier, lock = Barrier(expected_parallelism), Lock()
    active = peak = 0

    def generate(messages, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            # A whole group must enter concurrently; too few workers would time out.
            barrier.wait(timeout=5)
            return '{"present":true,"x":0.5,"y":0.5,"note":"scripted localization"}'
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(provider, "_generate", generate)
    result = provider.review("unused", (frame_count - 1) * 1000, "Where?", target="cup")
    assert peak == expected_parallelism and active == 0
    assert result["frames"] == evidence
    assert [r["at_ms"] for r in result["frame_reports"]] == [r["at_ms"] for r in evidence]
    assert result["provenance"]["review_workers"] == expected_parallelism
    assert not result["authoritative"]
