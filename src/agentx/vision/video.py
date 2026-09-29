import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration_ms: int
    fps: float


def probe(path: Path) -> VideoInfo:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("This file has no video stream.")
        stream = container.streams.video[0]
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration and stream.time_base
            else (container.duration or 0) / av.time_base
        )
        fps = float(stream.average_rate or 25)
        if duration <= 0 or stream.width < 16 or stream.height < 16:
            raise ValueError("The video must have readable dimensions and a finite duration.")
        if max(stream.width, stream.height) > 4096:
            raise ValueError("Videos above 4096 pixels per side are not supported by this MVP.")
        first = next(container.decode(stream), None)
        if first is None:
            raise ValueError("The video contains no decodable frames.")
        image = upright_image(first)
        height, width = image.shape[:2]
        return VideoInfo(width, height, round(duration * 1000), fps)


def upright_image(frame: av.VideoFrame) -> np.ndarray:
    """Match FFmpeg/browser orientation while preserving the source pixels."""
    image = frame.to_ndarray(format="bgr24")
    rotation = frame.rotation % 360
    if rotation == 0:
        return image
    rotations = {
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }
    if rotation not in rotations:
        raise ValueError("Video orientation must be a multiple of 90 degrees.")
    return cv2.rotate(image, rotations[rotation])


def frames(
    path: Path,
    sample_fps: float,
    start_ms: int = 0,
    end_ms: int | None = None,
    slack_ms: float = 0.5,
) -> Iterator[tuple[int, np.ndarray]]:
    """Forward-only sampling using source presentation timestamps, not frame counts.

    `slack_ms` is how early a frame may arrive and still count as the next sample. A file has
    regular timestamps; a live capture is stamped on arrival, so it passes a larger slack.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.time_base is None:
            raise ValueError("The video has no usable presentation time base.")
        time_base = stream.time_base
        origin = float((stream.start_time or 0) * time_base)
        next_sample = float(start_ms)
        last_ms = -1
        if start_ms > 1000:
            container.seek(
                int((origin + start_ms / 1000) / float(time_base)),
                stream=stream,
                backward=True,
            )
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            at_ms = round((float(frame.pts * frame.time_base) - origin) * 1000)
            if end_ms is not None and at_ms > end_ms:
                break
            if at_ms < next_sample - slack_ms or at_ms <= last_ms:
                continue
            last_ms = at_ms
            next_sample = at_ms + 1000 / sample_fps
            yield at_ms, upright_image(frame)


def read_frame(path: Path, at_ms: int) -> tuple[int, np.ndarray]:
    """Read the nearest source frame at or before the requested time."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.time_base is None:
            raise ValueError("The video has no usable presentation time base.")
        time_base = stream.time_base
        origin = float((stream.start_time or 0) * time_base)
        if at_ms > 0:
            container.seek(
                int((origin + at_ms / 1000) / float(time_base)), stream=stream, backward=True
            )
        best = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            timestamp = round((float(frame.pts * frame.time_base) - origin) * 1000)
            if timestamp > at_ms:
                break
            if timestamp >= 0:
                best = (timestamp, upright_image(frame))
        if best is None:
            raise ValueError("No decoded frame exists at or before this timestamp.")
        return best


def jpeg(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Could not encode evidence frame.")
    return encoded.tobytes()


def bounded_jpeg(frame: np.ndarray, max_edge: int) -> bytes:
    """JPEG with the longest edge at most max_edge. Reviews hash this exact byte stream, so
    every model input and its replay must go through this one function."""
    height, width = frame.shape[:2]
    scale = min(1, max_edge / max(height, width))
    if scale < 1:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    return jpeg(frame)


def ffmpeg_executable() -> str | None:
    """A system ffmpeg on PATH, else the optional static binary from imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def create_preview(source: Path, target: Path, timeout: int = 180) -> None:
    """Preserve the original separately; normalize only the browser playback proxy."""
    executable = ffmpeg_executable()
    if executable is None:
        raise ValueError("FFmpeg is not installed. Install ffmpeg or the ffmpeg extra.")
    result = subprocess.run(
        [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "setpts=PTS-STARTPTS,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ],
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise ValueError("The video could not be converted for browser playback.")
