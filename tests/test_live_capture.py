"""Live camera captures: memory follows a recording that is still growing."""

import hashlib
import threading
import time
from pathlib import Path

import av
import cv2
import pytest
from fastapi.testclient import TestClient

from agentx.api.app import create_app
from agentx.config import Settings
from agentx.services.demo import create_fixture
from agentx.services.live import LIVE_WARNING


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def camera_frames(tmp_path: Path):
    """The generated desk fixture, decoded, standing in for a camera at 15 FPS."""
    path = tmp_path / "camera.mp4"
    objects = create_fixture(path, "fixed")
    with av.open(str(path)) as container:
        images = [f.to_ndarray(format="bgr24") for f in container.decode(video=0)]
    return images, objects


def encoded(image) -> bytes:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return data.tobytes()


@pytest.fixture
def live_app(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url="",
        api_token="",
        stepfun_api_key="",
        stepfun_model="",
        cosmos_base_url="",
        worker_enabled=False,
        live_idle_seal_seconds=600,
        web_dist=Path("not-built"),
        _env_file=None,
    )
    app = create_app(settings)
    app.state.services.live.clock = FakeClock()
    return app


def push(client, app, video_id, images, step=0.2):
    clock = app.state.services.live.clock
    results = []
    for image in images:
        clock.now += step
        response = client.post(
            f"/api/v1/live/{video_id}/frames",
            content=encoded(image),
            headers={"content-type": "image/jpeg"},
        )
        assert response.status_code == 200, response.text
        results.append(response.json())
    return results


def wait_for(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def test_memory_follows_a_growing_capture_and_seals_it(live_app, tmp_path):
    images, objects = camera_frames(tmp_path)
    sampled = images[::3]  # 15 FPS camera, 5 FPS capture
    with TestClient(live_app) as client:
        video = client.post("/api/v1/live", json={"title": "Bench cam", "fps": 5}).json()
        assert video["live_status"] == "recording" and video["sha256"] == ""
        vid = video["id"]
        first = push(client, live_app, vid, sampled[:6])
        assert [r["at_ms"] for r in first] == [0, 200, 400, 600, 800, 1000]
        # One-frame latency: everything before the newest frame is readable.
        assert first[-1]["duration_ms"] == 801
        frame = client.get(f"/api/v1/videos/{vid}/frame?at_ms=0")
        assert frame.status_code == 200 and frame.headers["x-frame-time-ms"] == "0"

        for obj in objects:
            payload = obj.model_dump()
            payload["at_ms"] = 0
            created = client.post(f"/api/v1/videos/{vid}/objects", json=payload)
            assert created.status_code == 201, created.text
        run = client.post(f"/api/v1/videos/{vid}/runs", json={"backend": "reference"}).json()
        indexer = live_app.state.services.indexer
        worker = threading.Thread(target=indexer.process, args=(run["id"],))
        worker.start()

        push(client, live_app, vid, sampled[6:24])
        progressed = wait_for(
            lambda: (
                (r := client.get(f"/api/v1/runs/{run['id']}").json())["processed_ms"] >= 4000 and r
            )
        )
        assert progressed["status"] == "running"
        assert progressed["provenance"]["input_sha256"] is None
        name = objects[0].name
        answer = client.post(
            f"/api/v1/runs/{run['id']}/questions",
            json={"text": f"Where is {name}?", "use_provider": False},
        )
        assert answer.status_code == 200, answer.text
        body = answer.json()
        assert LIVE_WARNING in body["warnings"]
        assert body["as_of_ms"] <= progressed["processed_ms"] + 200
        assert body["evidence"], body
        evidence = client.get(body["evidence"][0]["frame_url"])
        assert evidence.status_code == 200
        review = client.post(
            f"/api/v1/runs/{run['id']}/review", json={"text": "What moved?", "at_ms": 1000}
        )
        assert review.status_code in {409}, review.text
        early = client.get(f"/api/v1/questions/{body['id']}/bundle")
        assert early.status_code == 422

        push(client, live_app, vid, sampled[24:])
        sealed = client.post(f"/api/v1/live/{vid}/stop")
        assert sealed.status_code == 200, sealed.text
        sealed = sealed.json()
        worker.join(timeout=60)
        assert not worker.is_alive()

        source = Path(live_app.state.services.settings.data_dir) / "videos" / vid / "source.mp4"
        assert sealed["live_status"] == "sealed"
        assert sealed["capture"]["first_frame_at"] and sealed["capture"]["encoder"] == "libx264"
        assert sealed["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert sealed["duration_ms"] >= (len(sampled) - 1) * 200
        assert client.get(sealed["media_url"]).status_code == 200
        done = client.get(f"/api/v1/runs/{run['id']}").json()
        assert done["status"] == "complete", done
        assert done["provenance"]["input_sha256"] == sealed["sha256"]
        assert done["processed_ms"] == (len(sampled) - 1) * 200
        # The answer asked while recording still revalidates against the sealed file.
        bundle = client.get(f"/api/v1/questions/{body['id']}/bundle")
        assert bundle.status_code == 200, bundle.text
        again = client.post(f"/api/v1/live/{vid}/frames", content=encoded(sampled[0]))
        assert again.status_code == 409

        # Memory built while recording equals an ordinary rebuild of the sealed file.
        rebuilt = client.post(f"/api/v1/videos/{vid}/runs", json={"backend": "reference"}).json()
        indexer.process(rebuilt["id"])

        def observations(run_id):
            rows = client.get(f"/api/v1/runs/{run_id}/export").json()["observations"]
            keys = ("object_id", "at_ms", "box", "zone", "score", "reason", "visible")
            return [tuple(str(row[k]) for k in keys) for row in rows]

        live_rows, offline_rows = observations(run["id"]), observations(rebuilt["id"])
        assert len(live_rows) == len(objects) * len(sampled)
        assert sorted(live_rows) == sorted(offline_rows)


def test_frames_faster_than_the_rate_are_dropped(live_app, tmp_path):
    images, _ = camera_frames(tmp_path)
    with TestClient(live_app) as client:
        vid = client.post("/api/v1/live", json={"fps": 5}).json()["id"]
        results = push(client, live_app, vid, images[:9], step=1 / 15)
        accepted = [r["at_ms"] for r in results if r["accepted"]]
        # A 15 FPS source becomes the requested 5 FPS; jitter up to a quarter interval is kept.
        assert accepted == [0, 200, 400]
        assert results[-1]["dropped"] == 6


def test_invalid_frames_and_stopping_empty_captures(live_app):
    with TestClient(live_app) as client:
        vid = client.post("/api/v1/live", json={}).json()["id"]
        bad = client.post(f"/api/v1/live/{vid}/frames", content=b"not an image")
        assert bad.status_code == 422
        stopped = client.post(f"/api/v1/live/{vid}/stop")
        assert stopped.status_code == 409
        assert client.get(f"/api/v1/videos/{vid}").status_code == 404


def test_restart_seals_a_capture_left_open(live_app, tmp_path):
    images, _ = camera_frames(tmp_path)
    with TestClient(live_app) as client:
        vid = client.post("/api/v1/live", json={}).json()["id"]
        push(client, live_app, vid, images[:30:3])
    # The first process exited without a stop: the capture is flushed but still "recording".
    settings = live_app.state.services.settings
    restarted = create_app(settings)
    with TestClient(restarted) as client:
        video = wait_for(
            lambda: (
                (v := client.get(f"/api/v1/videos/{vid}").json())["live_status"] == "sealed" and v
            )
        )
        source = settings.data_dir / "videos" / vid / "source.mp4"
        assert video["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert video["capture"]["sealed_reason"] == "recovered after restart"
        assert video["duration_ms"] >= 1800


def test_deleting_a_recording_capture_stops_it(live_app, tmp_path):
    images, _ = camera_frames(tmp_path)
    with TestClient(live_app) as client:
        vid = client.post("/api/v1/live", json={}).json()["id"]
        push(client, live_app, vid, images[:6:3])
        assert client.delete(f"/api/v1/videos/{vid}").status_code == 204
        after = client.post(f"/api/v1/live/{vid}/frames", content=encoded(images[0]))
        assert after.status_code == 404


def test_agent_token_cannot_start_or_feed_a_capture(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        api_token="operator",
        agent_token="agent",
        worker_enabled=False,
        web_dist=Path("not-built"),
        _env_file=None,
    )
    with TestClient(create_app(settings)) as client:
        headers = {"Authorization": "Bearer agent"}
        assert client.post("/api/v1/live", json={}, headers=headers).status_code == 403
        operator = {"Authorization": "Bearer operator"}
        assert client.post("/api/v1/live", json={}, headers=operator).status_code == 201


def test_live_capture_can_be_disabled(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        worker_enabled=False,
        live_enabled=False,
        web_dist=Path("not-built"),
        _env_file=None,
    )
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/v1/live", json={}).status_code == 404


def test_jittered_capture_with_late_registration_matches_its_rebuild(live_app, tmp_path):
    images, objects = camera_frames(tmp_path)
    sampled = images[::3]
    steps = [0.2, 0.17, 0.23, 0.16, 0.21, 0.19, 0.24, 0.18]  # arrival jitter around 5 FPS
    clock = live_app.state.services.live.clock
    with TestClient(live_app) as client:
        vid = client.post("/api/v1/live", json={"fps": 5}).json()["id"]
        stamps = []
        for index, image in enumerate(sampled):
            clock.now += steps[index % len(steps)]
            result = client.post(f"/api/v1/live/{vid}/frames", content=encoded(image)).json()
            assert result["accepted"], result
            stamps.append(result["at_ms"])
            if index == 8:
                # Register on a frame seen mid-capture, as a person watching the camera would.
                late = stamps[5]
                for obj in objects:
                    payload = obj.model_dump() | {"at_ms": late}
                    assert client.post(f"/api/v1/videos/{vid}/objects", json=payload).is_success
                run = client.post(f"/api/v1/videos/{vid}/runs", json={"backend": "reference"})
                run = run.json()
                indexer = live_app.state.services.indexer
                worker = threading.Thread(target=indexer.process, args=(run["id"],))
                worker.start()
        assert client.post(f"/api/v1/live/{vid}/stop").status_code == 200
        worker.join(timeout=60)
        rebuilt = client.post(f"/api/v1/videos/{vid}/runs", json={"backend": "reference"}).json()
        indexer.process(rebuilt["id"])

        def rows(run_id):
            data = client.get(f"/api/v1/runs/{run_id}/export").json()["observations"]
            keys = ("object_id", "at_ms", "box", "zone", "score", "reason", "visible")
            return sorted(tuple(str(row[k]) for k in keys) for row in data)

        live_rows = rows(run["id"])
        # Every captured frame from the registration onward was read; none skipped for jitter.
        assert sorted({int(r[1]) for r in live_rows}) == stamps[5:]
        assert live_rows == rows(rebuilt["id"])


def test_weight_cache_loads_once_and_rehashes_changed_files(tmp_path):
    from agentx.vision import weights

    calls = []
    first = weights.cached(("demo",), lambda: calls.append(1) or object())
    assert weights.cached(("demo",), lambda: calls.append(1) or object()) is first
    assert calls == [1]
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"one")
    before = weights.file_sha256(checkpoint)
    checkpoint.write_bytes(b"two!")
    assert weights.file_sha256(checkpoint) != before


def test_openh264_fallback_keeps_one_frame_latency(tmp_path, monkeypatch):
    from agentx.services import live

    real = av.codec.Codec

    def without_x264(name, mode="r"):
        if name == "libx264":
            raise ValueError("not in this build")
        return real(name, mode)

    monkeypatch.setattr(live.av.codec, "Codec", without_x264)
    images, _ = camera_frames(tmp_path)
    writer = live._Writer(tmp_path / "capture.mp4", 5, 1280)
    stamps = [writer.append(image, 10 + i * 0.2) for i, image in enumerate(images[:30:3])]
    assert writer.codec == "libopenh264"
    assert writer.readable_ms == stamps[-2]
    from agentx.vision.video import read_frame

    assert read_frame(tmp_path / "capture.mp4", stamps[-2])[0] == stamps[-2]
    writer.close()


def test_a_failed_playback_proxy_still_seals_and_finishes_memory(live_app, tmp_path, monkeypatch):
    from agentx.services import live

    def broken_preview(source, target, timeout=180):
        raise ValueError("FFmpeg is not installed.")

    monkeypatch.setattr(live, "create_preview", broken_preview)
    images, objects = camera_frames(tmp_path)
    with TestClient(live_app) as client:
        vid = client.post("/api/v1/live", json={}).json()["id"]
        push(client, live_app, vid, images[:18:3])
        payload = objects[0].model_dump() | {"at_ms": 0}
        assert client.post(f"/api/v1/videos/{vid}/objects", json=payload).is_success
        run = client.post(f"/api/v1/videos/{vid}/runs", json={"backend": "reference"}).json()
        worker = threading.Thread(target=live_app.state.services.indexer.process, args=(run["id"],))
        worker.start()
        sealed = client.post(f"/api/v1/live/{vid}/stop").json()
        worker.join(timeout=60)
        assert sealed["live_status"] == "sealed" and len(sealed["sha256"]) == 64
        assert sealed["capture"]["playback_error"].startswith("Playback media could not")
        assert client.get(f"/api/v1/runs/{run['id']}").json()["status"] == "complete"


def test_deleting_during_a_seal_is_refused(live_app, tmp_path):
    images, _ = camera_frames(tmp_path)
    with TestClient(live_app) as client:
        vid = client.post("/api/v1/live", json={}).json()["id"]
        push(client, live_app, vid, images[:9:3])
        live_app.state.services.live.sealing.add(vid)
        assert client.delete(f"/api/v1/videos/{vid}").status_code == 409
        live_app.state.services.live.sealing.discard(vid)
        assert client.delete(f"/api/v1/videos/{vid}").status_code == 204


def test_oversized_frames_are_refused_before_decoding(live_app):
    import numpy as np

    with TestClient(live_app) as client:
        vid = client.post("/api/v1/live", json={}).json()["id"]
        ok, png = cv2.imencode(".png", np.zeros((16, 5000, 3), np.uint8))
        response = client.post(f"/api/v1/live/{vid}/frames", content=png.tobytes())
        assert response.status_code == 422
        assert "4096 pixels" in response.json()["detail"]


def test_a_capture_cut_mid_fragment_is_recovered_and_rebuilt(live_app, tmp_path):
    images, objects = camera_frames(tmp_path)
    settings = live_app.state.services.settings
    with TestClient(live_app) as client:
        vid = client.post("/api/v1/live", json={}).json()["id"]
        stamps = [r["at_ms"] for r in push(client, live_app, vid, images[:36:3])]
    source = settings.data_dir / "videos" / vid / "source.mp4"
    with source.open("r+b") as stream:  # a crash while the last fragment was being written
        stream.truncate(source.stat().st_size - 700)
    restarted = create_app(settings)
    with TestClient(restarted) as client:
        video = wait_for(
            lambda: (
                (v := client.get(f"/api/v1/videos/{vid}").json())["live_status"] == "sealed" and v
            )
        )
        assert stamps[-3] < video["duration_ms"] <= stamps[-1] + 1
        payload = objects[0].model_dump() | {"at_ms": 0}
        assert client.post(f"/api/v1/videos/{vid}/objects", json=payload).is_success
        run = client.post(f"/api/v1/videos/{vid}/runs", json={"backend": "reference"}).json()
        restarted.state.services.indexer.process(run["id"])
        assert client.get(f"/api/v1/runs/{run['id']}").json()["status"] == "complete"


def test_decoder_errors_do_not_reveal_storage_paths(live_app, tmp_path):
    images, _ = camera_frames(tmp_path)
    with TestClient(live_app) as client:
        vid = client.post("/api/v1/live", json={}).json()["id"]
        push(client, live_app, vid, images[:12:3])
        source = live_app.state.services.settings.data_dir / "videos" / vid / "source.mp4"
        source.write_bytes(b"\x00" * 64)
        response = client.get(f"/api/v1/videos/{vid}/frame?at_ms=0")
        assert response.status_code == 422
        assert str(source.parent) not in response.text and "videos/" not in response.text
