"""Generated, explicitly labeled software fixtures. Never a real-world accuracy benchmark.

Two scenes are available. `fixed` is the original stationary camera. `bumped` keeps every object
exactly where it was and moves the *camera* instead, which is the case a fixed-camera product has
to survive: the pixels of a resting object move, but the place it is resting in does not change.
"""

from pathlib import Path
from typing import Literal

import av
import cv2
import numpy as np

from agentx.domain.contracts import Box, RegisterObject

Scene = Literal["fixed", "bumped"]
SCENES: dict[str, str] = {
    "fixed": "Stationary camera: one object moves, is occluded, then settles in the centre.",
    "bumped": "The camera is knocked at six seconds and never straightened. Nothing on the desk "
    "moves afterwards, so every region answer must stay what it was before the knock.",
}
# The knock: a pan-and-zoom applied to the canvas viewport, eased over 1.5 s and never undone.
BUMP_START, BUMP_SECONDS = 6.0, 1.5
# Large enough that both resting objects appear in a different third of the moved frame:
# reading the pixels naively gives the wrong region, which is the point of the scene.
BUMP_SHIFT, BUMP_RISE, BUMP_ZOOM = 200.0, 14.0, 0.07


def _viewport(t: float, scene: Scene, canvas: tuple[int, int]) -> tuple[float, float, float]:
    canvas_width, canvas_height = canvas
    left, top = (canvas_width - 640) / 2, (canvas_height - 360) / 2
    if scene == "fixed" or t < BUMP_START:
        return left, top, 1.0
    progress = min(1.0, (t - BUMP_START) / BUMP_SECONDS)
    eased = progress * progress * (3 - 2 * progress)
    return left + BUMP_SHIFT * eased, top + BUMP_RISE * eased, 1 + BUMP_ZOOM * eased


def _view(canvas_frame: np.ndarray, viewport: tuple[float, float, float]) -> np.ndarray:
    left, top, scale = viewport
    matrix = np.array([[scale, 0, -left * scale], [0, scale, -top * scale]], dtype=np.float32)
    return cv2.warpAffine(
        canvas_frame, matrix, (640, 360), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def create_fixture(path: Path, scene: Scene = "fixed") -> list[RegisterObject]:
    width, height, fps, seconds = 640, 360, 15, 12
    # A knocked camera must reveal real scene, so the bumped variant paints a larger desk and
    # views a moving window into it. The first frame is identical in both scenes.
    canvas_width = round(width * 1.8) if scene == "bumped" else width
    canvas_height = round(height * 1.8) if scene == "bumped" else height
    dx, dy = (canvas_width - width) // 2, (canvas_height - height) // 2
    rng = np.random.default_rng(27)
    background = np.full((canvas_height, canvas_width, 3), (32, 42, 37), dtype=np.uint8)
    noise = rng.integers(-5, 6, background.shape, dtype=np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for x in (213, 426):
        cv2.line(background, (x + dx, 65 + dy), (x + dx, 340 + dy), (69, 85, 76), 1)
    for x, text in ((80, "LEFT"), (280, "CENTER"), (500, "RIGHT")):
        cv2.putText(
            background,
            text,
            (x + dx, 330 + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (128, 151, 137),
            1,
        )
    cv2.putText(
        background,
        "CONTROLLED FIXTURE - NOT A MODEL BENCHMARK",
        (28 + dx, 30 + dy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 194, 184),
        1,
    )
    if scene == "bumped":
        # Texture the margin the knock reveals, and a desk edge the estimator can hold on to.
        cv2.rectangle(background, (dx - 96, dy + 60), (dx - 24, dy + 300), (58, 74, 66), -1)
        cv2.rectangle(
            background, (dx + width + 22, dy + 40), (dx + width + 104, dy + 250), (44, 58, 52), -1
        )
        for i in range(0, canvas_height, 37):
            cv2.line(background, (0, i), (canvas_width, i + 7), (46, 58, 52), 1)
        cv2.putText(
            background,
            "CAMERA KNOCKED AT 00:06 - OBJECTS DO NOT MOVE",
            (28 + dx, 52 + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (150, 180, 200),
            1,
        )

    def patch(color, name, seed):
        image = np.full((64, 90, 3), color, dtype=np.uint8)
        noise_rng = np.random.default_rng(seed)
        for _ in range(45):
            x, y = noise_rng.integers(5, 85), noise_rng.integers(5, 59)
            cv2.circle(image, (int(x), int(y)), int(noise_rng.integers(1, 4)), (30, 30, 30), -1)
        cv2.rectangle(image, (1, 1), (88, 62), (230, 230, 230), 2)
        cv2.putText(image, name, (7, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
        return image

    red = patch((65, 73, 205), "TOOLKIT", 4)
    blue = patch((181, 103, 43), "REMOTE", 19)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        for i in range(fps * seconds):
            t = i / fps
            frame = background.copy()
            if scene == "bumped":
                # Only the camera moves after six seconds. The toolkit settles on the right
                # beforehand and stays there, so its region must survive the knock.
                x = 65 if t < 3 else 480
                frame[125 + dy : 189 + dy, x + dx : x + 90 + dx] = red
            else:
                if t < 3:
                    x = 65
                elif t < 6:
                    x = 480
                else:
                    x = 280
                if not 6 <= t < 8:
                    frame[125:189, x : x + 90] = red
                else:
                    cv2.rectangle(frame, (445, 90), (605, 205), (161, 172, 160), -1)
                    cv2.putText(
                        frame,
                        "OCCLUDED",
                        (465, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (50, 60, 50),
                        1,
                    )
            frame[235 + dy : 299 + dy, 340 + dx : 430 + dx] = blue
            if scene == "bumped":
                frame = _view(frame, _viewport(t, scene, (canvas_width, canvas_height)))
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return [
        RegisterObject(
            name="Red toolkit",
            label="custom",
            box=Box(x1=65 / width, y1=125 / height, x2=155 / width, y2=189 / height),
        ),
        RegisterObject(
            name="Blue remote",
            label="remote",
            box=Box(x1=340 / width, y1=235 / height, x2=430 / width, y2=299 / height),
        ),
    ]
