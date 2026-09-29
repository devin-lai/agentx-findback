"""Camera captures that memory can follow while they are still being recorded.

A live recording is an ordinary source video that is allowed to grow. Frames arrive one at a
time (from a browser camera or `agentx live-push`), are encoded once into a fragmented MP4 and
become readable as soon as their fragment is on disk. The indexer decodes the same file, so the
pixels memory was built from are exactly the pixels an evidence frame replays.

While a capture is recording it has no final SHA-256, so answers carry a warning and evidence
reports are refused. Stopping the capture seals it: the file is hashed, a playback proxy is
made, and from then on it is indistinguishable from an uploaded recording.
"""

import hashlib
import io
import logging
import shutil
import threading
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from PIL import Image
from sqlalchemy import select

from agentx.domain.contracts import default_regions
from agentx.storage.models import IndexRun, Video, uid
from agentx.vision.video import create_preview, jpeg, probe, read_frame

logger = logging.getLogger(__name__)

RECORDING = "recording"
MAX_FRAME_EDGE = 4096
SEALED = "sealed"
LIVE_WARNING = (
    "This answer comes from a live recording that is still growing. Its evidence frames are "
    "read from the capture file; the file is hashed when the recording stops."
)


def decodable_end_ms(source: Path) -> int:
    """One past the last frame that decodes, tolerating a tail cut mid-fragment."""
    last = -1
    try:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            origin = float((stream.start_time or 0) * (stream.time_base or 0))
            for frame in container.decode(stream):
                if frame.pts is not None and frame.time_base is not None:
                    last = round((float(frame.pts * frame.time_base) - origin) * 1000)
    except (av.error.FFmpegError, ValueError) as exc:
        logger.info("Capture %s ends in an incomplete fragment: %s", source.parent.name, exc)
    return last + 1


class LiveClosed(ValueError):
    """Frames for a capture that has already been sealed or never existed here."""


def _even(value: int) -> int:
    return max(16, value - value % 2)


def encoder() -> str:
    """libx264 where the installed PyAV build has it (GPL component); otherwise the BSD-licensed
    OpenH264 encoder that the same binary wheels also ship."""
    for name in ("libx264", "libopenh264"):
        try:
            av.codec.Codec(name, "w")
            return name
        except (av.error.FFmpegError, ValueError):
            continue
    raise ValueError("Live capture needs an H.264 encoder (libx264 or libopenh264) in PyAV.")


class _Writer:
    """One fragmented H.264 file. Each frame gets its own fragment, so everything before the
    most recent frame is readable by another process while the file keeps growing."""

    def __init__(self, path: Path, fps: float, max_edge: int):
        self.path, self.fps, self.max_edge = path, fps, max_edge
        self.codec = ""
        self.first_frame_at: str | None = None
        self.container: Any = None
        self.stream: Any = None
        self.size: tuple[int, int] = (0, 0)
        self.started = 0.0
        self.last_pts = -1
        self.readable_ms = -1
        self.frames = 0
        self.dropped = 0
        self.last_frame_at = time.monotonic()
        self.closed = False
        self.lock = threading.Lock()

    def _open(self, width: int, height: int, now: float):
        scale = min(1.0, self.max_edge / max(width, height))
        self.size = (_even(round(width * scale)), _even(round(height * scale)))
        codec = encoder()
        self.container = av.open(
            str(self.path),
            "w",
            format="mp4",
            options={
                "movflags": "frag_every_frame+empty_moov+default_base_moof",
                "flush_packets": "1",
            },
        )
        self.codec = codec
        stream = self.container.add_stream(self.codec, rate=Fraction(round(self.fps * 1000), 1000))
        stream.width, stream.height = self.size
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        gop = str(max(1, round(self.fps * 2)))
        # zerolatency keeps the encoder from holding frames back; a two-second GOP bounds seeks.
        stream.options = (
            {"preset": "veryfast", "tune": "zerolatency", "crf": "23", "g": gop, "keyint_min": gop}
            if self.codec == "libx264"
            else {"b": "4000000", "g": gop}
        )
        self.stream = stream
        self.started = now
        # Wall-clock time of the capture's 0 ms, so a client can show "last seen at 14:02:31".
        self.first_frame_at = datetime.now(UTC).isoformat()

    def append(self, image: np.ndarray, now: float) -> int | None:
        """Encode one frame at its arrival time; None when it arrived faster than the rate."""
        with self.lock:
            if self.closed:
                raise LiveClosed("This live recording has stopped. Start a new capture.")
            if self.container is None:
                self._open(image.shape[1], image.shape[0], now)
            pts = round((now - self.started) * 1000)
            if self.last_pts >= 0 and pts - self.last_pts < 1000 / self.fps * 0.75:
                self.dropped += 1
                return None
            pts = max(pts, self.last_pts + 1)
            if (image.shape[1], image.shape[0]) != self.size:
                image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="bgr24")
            frame.pts = pts
            frame.time_base = Fraction(1, 1000)
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            # The fragment holding the previous frame is complete once this one is muxed.
            self.readable_ms = self.last_pts
            self.last_pts = pts
            self.frames += 1
            self.last_frame_at = now
            return pts

    def close(self):
        with self.lock:
            self.closed = True
            if self.container is None:
                return
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
                self.container = None
                self.readable_ms = self.last_pts


class LiveRecorder:
    def __init__(self, database, settings, catalog):
        self.db, self.settings, self.catalog = database, settings, catalog
        self.writers: dict[str, _Writer] = {}
        self.lock = threading.Lock()
        self.seal_lock = threading.Lock()
        self.sealing: set[str] = set()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        # Arrival time is the capture's clock; tests substitute a deterministic one.
        self.clock = time.monotonic

    # Lifecycle -----------------------------------------------------------------------------

    def start(self):
        """Watch for abandoned captures; seal any that a previous process left open."""
        self.thread = threading.Thread(target=self._watch, name="agentx-live", daemon=True)
        self.thread.start()

    def close(self):
        """Flush open captures without sealing them; the next start seals what is on disk.

        Sealing makes a playback proxy, which can take far longer than a graceful shutdown.
        """
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
        with self.lock:
            writers, self.writers = list(self.writers.values()), {}
        for writer in writers:
            try:
                writer.close()
            except Exception:
                logger.exception("Could not flush a live recording on shutdown")

    def _orphans(self) -> list[str]:
        with self.db.session() as session:
            ids = session.scalars(select(Video.id).where(Video.live_status == RECORDING)).all()
        return [video_id for video_id in ids if video_id not in self.writers]

    def _watch(self):
        for video_id in self._orphans():
            try:
                self.seal(video_id, reason="recovered after restart")
            except Exception:
                logger.exception("Could not seal orphaned live recording %s", video_id)
        while not self.stop.wait(2):
            idle = self.settings.live_idle_seal_seconds
            now = self.clock()
            for video_id, writer in list(self.writers.items()):
                if now - writer.last_frame_at > idle:
                    logger.info(
                        "Sealing live recording %s after %ss without frames", video_id, idle
                    )
                    try:
                        self.seal(video_id, reason=f"no frames for {idle} s")
                    except Exception:
                        logger.exception("Could not seal idle live recording %s", video_id)

    # Capture -------------------------------------------------------------------------------

    def create(self, title: str, fps: float, source: str) -> Video:
        video_id = uid()
        directory = self.settings.data_dir / "videos" / video_id
        directory.mkdir(parents=True)
        source_path = directory / "source.mp4"
        writer = _Writer(source_path, fps, self.settings.live_max_edge)
        writer.last_frame_at = self.clock()
        clean = " ".join(title.split())[:120] or "Live camera"
        video = Video(
            id=video_id,
            title=clean,
            original_name=f"{clean[:100]}.mp4",
            sha256="",
            duration_ms=0,
            width=0,
            height=0,
            fps=fps,
            source_path=str(source_path.relative_to(self.settings.data_dir)),
            media_path=str((directory / "playback.mp4").relative_to(self.settings.data_dir)),
            regions=default_regions(),
            is_fixture=False,
            live_status=RECORDING,
            capture={
                "source": source,
                "fps": fps,
                "started_at": datetime.now(UTC).isoformat(),
                "frames": 0,
                "dropped": 0,
            },
        )
        with self.lock:
            self.writers[video_id] = writer
        with self.db.session.begin() as session:
            session.add(video)
        return video

    def append(self, video_id: str, encoded: bytes) -> dict:
        writer = self.writers.get(video_id)
        if writer is None:
            with self.db.session() as session:
                video = session.get(Video, video_id)
            if video is None:
                raise LookupError("Video not found.")
            raise LiveClosed("This live recording has stopped. Start a new capture.")
        if len(encoded) > self.settings.live_max_frame_bytes:
            raise ValueError("A live frame exceeds the per-frame size limit.")
        # Read the declared size before decoding: a small compressed image can expand to gigabytes.
        try:
            with Image.open(io.BytesIO(encoded)) as header:
                width, height = header.size
        except Exception:
            raise ValueError("Send each live frame as a JPEG or PNG image.") from None
        if max(width, height) > MAX_FRAME_EDGE:
            raise ValueError(f"Live frames must be at most {MAX_FRAME_EDGE} pixels per side.")
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < 16:
            raise ValueError("Send each live frame as a JPEG or PNG image.")
        pts = writer.append(image, self.clock())
        if pts is not None and pts >= self.settings.max_video_seconds * 1000:
            self.seal(video_id, reason="maximum duration reached")
            raise LiveClosed("The live recording reached its maximum duration and was sealed.")
        readable = writer.readable_ms
        if pts is not None and readable >= 0:
            with self.db.session.begin() as session:
                video = session.get(Video, video_id)
                if video is not None and video.live_status == RECORDING:
                    video.duration_ms = readable + 1
                    video.width, video.height = writer.size
                    video.capture = {
                        **(video.capture or {}),
                        "frames": writer.frames,
                        "dropped": writer.dropped,
                        "encoder": writer.codec,
                        "first_frame_at": writer.first_frame_at,
                    }
        return {
            "accepted": pts is not None,
            "at_ms": pts,
            "duration_ms": readable + 1 if readable >= 0 else 0,
            "frames": writer.frames,
            "dropped": writer.dropped,
        }

    def seal(self, video_id: str, reason: str = "stopped by operator") -> Video:
        """Hash the capture and make it an ordinary recording. One seal runs at a time, so a
        second stop (or the idle watcher racing a stop) sees the sealed result."""
        with self.seal_lock:
            self.sealing.add(video_id)
            try:
                return self._seal(video_id, reason)
            finally:
                self.sealing.discard(video_id)

    def is_sealing(self, video_id: str) -> bool:
        return video_id in self.sealing

    def _seal(self, video_id: str, reason: str) -> Video:
        with self.lock:
            writer = self.writers.pop(video_id, None)
        if writer is not None:
            writer.close()
        with self.db.session() as session:
            video = session.get(Video, video_id)
        if video is None:
            raise LookupError("Video not found.")
        if video.live_status != RECORDING:
            if writer is None:
                raise LiveClosed("This live recording is already sealed.")
            return video
        source = self.catalog.path(video.source_path)
        if not source.is_file() or source.stat().st_size == 0 or (writer and writer.frames < 2):
            self._discard(video_id)
            raise LiveClosed("The live recording stopped before it captured any usable frames.")
        with source.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        width, height = video.width, video.height
        if writer is None:
            # Recovered after a crash: the tail may end mid-fragment, so the duration is the
            # last frame that actually decodes, not what the container header claims.
            duration = decodable_end_ms(source)
        else:
            try:
                info = probe(source)
                duration, width, height = info.duration_ms, info.width, info.height
            except (ValueError, av.FFmpegError):
                duration = video.duration_ms
        with self.db.session.begin() as session:
            current = session.get(Video, video_id)
            if current is None:
                raise LookupError("Video not found.")
            current.sha256 = checksum
            current.duration_ms = max(duration, 1)
            current.width, current.height = width, height
            current.live_status = SEALED
            current.capture = {
                **(current.capture or {}),
                "sealed_at": datetime.now(UTC).isoformat(),
                "sealed_reason": reason,
                **({"frames": writer.frames, "dropped": writer.dropped} if writer else {}),
            }
        # The capture is sealed and hashed; a missing playback proxy must not undo that.
        directory = source.parent
        problem = None
        try:
            create_preview(source, directory / "playback.mp4")
            _, first = read_frame(source, 0)
            (directory / "poster.jpg").write_bytes(jpeg(first))
        except Exception as exc:
            logger.exception("Could not create playback media for sealed capture %s", video_id)
            problem = f"Playback media could not be created: {type(exc).__name__}"
        with self.db.session.begin() as session:
            current = session.get(Video, video_id)
            if current is None:
                raise LookupError("Video not found.")
            if problem:
                current.capture = {**(current.capture or {}), "playback_error": problem}
            return current

    def _discard(self, video_id: str):
        with self.db.session.begin() as session:
            if session.scalar(select(IndexRun.id).where(IndexRun.video_id == video_id)):
                session.get(Video, video_id).live_status = SEALED
                return
            session.delete(session.get(Video, video_id))
        shutil.rmtree(self.settings.data_dir / "videos" / video_id, ignore_errors=True)

    def discard(self, video_id: str):
        """Stop accepting frames for a capture that is about to be deleted."""
        with self.lock:
            writer = self.writers.pop(video_id, None)
        if writer is not None:
            writer.close()
