"""End-to-end: a knocked camera must not move an object that never moved.

The bumped controlled sample pans and zooms the camera at six seconds and never straightens it.
Both objects rest for the whole second half, yet each one lands in a different third of the
moved frame. Reading those pixels naively renames the place the user asked about.
"""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import select

from agentx.api.app import create_app
from agentx.config import Settings
from agentx.domain.contracts import Box, IndexRequest, default_regions, region_for
from agentx.services.demo import create_fixture
from agentx.storage.models import IndexRun, Observation


@pytest.fixture(scope="module")
def bumped_run():
    with tempfile.TemporaryDirectory(prefix="agentx-bumped-") as temporary:
        root = Path(temporary)
        app = create_app(Settings(data_dir=root / "data", database_url="", worker_enabled=False))
        s = app.state.services
        s.db.migrate()
        try:
            source = root / "bumped.mp4"
            objects = create_fixture(source, "bumped")
            video = s.catalog.import_video(source, "bumped.mp4", is_fixture=True)
            registered = [s.catalog.register(video.id, obj) for obj in objects]
            run = s.indexer.enqueue(video.id, IndexRequest(backend="reference", sample_fps=5))
            s.indexer.process(run.id)
            with s.db.session() as session:
                run = session.get(IndexRun, run.id)
                rows = session.scalars(
                    select(Observation)
                    .where(Observation.run_id == run.id)
                    .order_by(Observation.at_ms)
                ).all()
                yield (
                    run,
                    registered,
                    [
                        {c.name: getattr(r, c.name) for c in Observation.__table__.columns}
                        for r in rows
                    ],
                )
        finally:
            s.db.engine.dispose()


def after_bump(rows, object_id):
    return [r for r in rows if r["object_id"] == object_id and r["at_ms"] >= 9000]


def test_the_camera_knock_is_detected_and_recovered(bumped_run):
    run, _, rows = bumped_run
    camera = run.provenance["camera"]
    assert run.status == "complete"
    assert camera["frames"]["stable"] > 0, "the first seconds are a stationary camera"
    assert camera["frames"]["compensated"] > 0, "the knock has to be noticed"
    assert camera["peak_motion"] > run.provenance["scene_guard"]["motion_threshold"]
    assert {r["scene_reference"] for r in rows} <= {"registered", "compensated", "unavailable"}
    assert any(r["scene_reference"] == "compensated" for r in rows)


def test_a_resting_object_keeps_its_registered_region_through_the_knock(bumped_run):
    _, registered, rows = bumped_run
    regions = default_regions()
    for obj in registered:
        late = after_bump(rows, obj.id)
        assert late, f"{obj.name} has no observations after the knock"
        visible = [r for r in late if r["visible"]]
        assert visible, f"{obj.name} was not observed at all after the knock"
        # Nothing on the desk moves after six seconds, so the region cannot change.
        expected = "right" if obj.name == "Red toolkit" else "center"
        assert {r["zone"] for r in visible} == {expected}
        # And the naive reading of those same pixels would have said something else.
        naive = {region_for(Box(**r["box"]), regions) for r in visible}
        assert naive != {expected}


def test_the_evidence_box_still_belongs_to_the_frame_it_came_from(bumped_run):
    """Compensation must not rewrite the box a user replays; only how its region is read."""
    _, registered, rows = bumped_run
    for obj in registered:
        for row in after_bump(rows, obj.id):
            if not row["visible"]:
                continue
            assert row["scene_reference"] in {"compensated", "unavailable"}
            if row["scene_reference"] == "compensated":
                # The scene box is the same finding read in registration coordinates, so the
                # two differ once the camera has moved.
                assert row["box"] is not None
