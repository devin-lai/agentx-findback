"""Explicit reference-image coordinate contract; public points remain in 0..1."""

import copy
import json

COORDINATE_RULE = (
    "Use integer x and y between 0 and 1000, with each source-image axis independently "
    "normalized: (0,0) is top-left and (1000,1000) is bottom-right. Do not use decimals or pixels. "
)
ROLE_RULE = (
    " Image 1 is only an identity reference crop. It is NOT evidence of "
    "presence. Image 2 is the complete source frame to inspect. "
    "Every coordinate must refer to image 2, NEVER to the reference crop. "
    + COORDINATE_RULE
    + "If the same-looking objects cannot be distinguished, report present=false; do not pick "
    "one arbitrarily. Keep note short and do not transcribe writing on objects."
)


def integer_reference_system(system: str) -> str:
    """Match the frozen, separately evaluated integer-points research prompt."""
    old = (
        "x and y are the normalized center of the target in the frame (0,0 is the top-left "
        "corner; 1,1 is the bottom-right corner), or null when it is not visible. "
    )
    if old not in system:
        raise ValueError("The base coordinate prompt changed; review the integer contract.")
    system = system.replace(
        old, COORDINATE_RULE + "Use null coordinates when the target is not visible. "
    )
    prefix, schema_text = system.split("JSON schema: ", 1)
    schema_line, suffix = schema_text.split("\n", 1)
    schema = json.loads(schema_line)
    for axis in ("x", "y"):
        schema["properties"][axis]["anyOf"] = [
            {"type": "integer", "minimum": 0, "maximum": 1000},
            {"type": "null"},
        ]
    return prefix + "JSON schema: " + json.dumps(schema) + "\n" + suffix + ROLE_RULE


def integer_reference_content(content: list[dict]) -> list[dict]:
    """Preserve source/reference bytes and frame identity while declaring their roles."""
    content = copy.deepcopy(content)
    images = [part for part in content if part["type"] == "image_url"]
    if len(images) != 2:
        raise ValueError("Integer reference grounding requires a reference and a source frame.")
    frame_label = next(p for p in content if p.get("text", "").startswith("frame_id="))
    target = next(p["text"] for p in content if p.get("text", "").startswith("Target object"))
    return [
        {"type": "text", "text": "IDENTITY REFERENCE ONLY (not the source frame):"},
        images[0],
        {"type": "text", "text": "SOURCE FRAME: report coordinates in this image only."},
        frame_label,
        images[1],
        {"type": "text", "text": target + ROLE_RULE},
    ]


def normalize_integer_report(report: dict) -> dict:
    """Strict declared units, never guessed from magnitudes, labels or frame size."""
    if set(report) != {"present", "x", "y", "note"} or type(report["present"]) is not bool:
        raise ValueError("Invalid integer-coordinate response schema.")
    if not isinstance(report["note"], str):
        raise ValueError("A localization note must be text.")
    normalized = report.copy()
    for axis in ("x", "y"):
        value = report[axis]
        if report["present"]:
            if type(value) is not int or not 0 <= value <= 1000:
                raise ValueError("Visible coordinates must be integers in 0..1000.")
            normalized[axis] = value / 1000
        elif value is not None:
            raise ValueError("Absent coordinates must be null.")
    return normalized
