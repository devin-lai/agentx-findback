import subprocess

import numpy as np
import pytest

from agentx.services.demo import create_fixture
from agentx.vision.video import ffmpeg_executable, frames, read_frame


@pytest.mark.parametrize("rotation", [90, 180, -90])
def test_phone_rotation_matches_playback_and_evidence(app, rotation, tmp_path):
    source = tmp_path / "landscape.mp4"
    create_fixture(source)
    rotated = tmp_path / "phone.mp4"
    executable = ffmpeg_executable()
    assert executable is not None, "Orientation tests require the supported FFmpeg runtime."
    subprocess.run(
        [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-display_rotation",
            str(rotation),
            "-i",
            str(source),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    s = app.state.services
    s.db.migrate()
    try:
        video = s.catalog.import_video(rotated, "phone.mp4")
        expected = (640, 360) if rotation == 180 else (360, 640)
        assert (video.width, video.height) == expected
        original_path = s.catalog.path(video.source_path)
        proxy_path = s.catalog.path(video.media_path)
        observed_at, evidence = read_frame(original_path, 2000)
        played_at, playback = read_frame(proxy_path, 2000)
        assert observed_at == played_at == 2000
        assert evidence.shape == playback.shape
        # Proxy compression may change colors; the orientation and object layout must match.
        assert np.abs(evidence.astype(float) - playback.astype(float)).mean() < 8
        sampled_at, sampled = next(frames(original_path, 5, start_ms=2000))
        assert sampled_at == 2000
        assert np.array_equal(sampled, evidence)
    finally:
        s.db.engine.dispose()
