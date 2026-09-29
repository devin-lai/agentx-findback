"""Score the existing Cosmos frame-localization contract on frozen real-video windows.

Point-in-box is weaker than box IoU or independently verified physical identity.
All failed windows stay in the denominator; these are correlated research frames.
"""

import argparse
import copy
import hashlib
import json
import math
import platform
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from agentx.agents.providers import Providers, extract_json_object
from agentx.agents.skills import SkillRegistry
from agentx.config import Settings
from agentx.domain.contracts import default_regions
from agentx.vision.video import jpeg, read_frame


def variant_messages(messages, variant):
    """Research-only image-role ablations. Production providers remain unchanged."""
    if variant == "deployed":
        return messages
    modified = copy.deepcopy(messages)
    content = modified[1]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    if len(images) != 2:
        raise ValueError("Image-role experiments require a reference and a source frame.")
    frame_label = next(p for p in content if p.get("text", "").startswith("frame_id="))
    target = next(p["text"] for p in content if p.get("text", "").startswith("Target object"))
    first_scene = variant == "scene_first"
    scene_number = 1 if first_scene else 2
    reference_number = 2 if first_scene else 1
    coordinate_rule = (
        "Use integer x and y between 0 and 1000, with each source-image axis independently "
        "normalized: (0,0) is top-left and (1000,1000) is bottom-right. Do not use decimals or pixels. "
        if variant == "integer_points"
        else "Use normalized decimals between 0 and 1 for x and y, not pixel values or 0-1000 units. "
    )
    if variant == "integer_points":
        old = (
            "x and y are the normalized center of the target in the frame (0,0 is the top-left "
            "corner; 1,1 is the bottom-right corner), or null when it is not visible. "
        )
        if old not in modified[0]["content"]:
            raise ValueError("The base coordinate prompt changed; review the adapter.")
        modified[0]["content"] = modified[0]["content"].replace(
            old, coordinate_rule + "Use null coordinates when the target is not visible. "
        )
        prefix, schema_text = modified[0]["content"].split("JSON schema: ", 1)
        schema_line, suffix = schema_text.split("\n", 1)
        schema = json.loads(schema_line)
        for axis in ("x", "y"):
            schema["properties"][axis]["anyOf"] = [
                {"type": "integer", "minimum": 0, "maximum": 1000},
                {"type": "null"},
            ]
        modified[0]["content"] = prefix + "JSON schema: " + json.dumps(schema) + "\n" + suffix
    role_rule = (
        f" Image {reference_number} is only an identity reference crop. It is NOT evidence of "
        f"presence. Image {scene_number} is the complete source frame to inspect. "
        f"Every coordinate must refer to image {scene_number}, NEVER to the reference crop. "
        + coordinate_rule
        + "If the same-looking objects cannot be distinguished, report present=false; do not pick "
        "one arbitrarily. Keep note short and do not transcribe writing on objects."
    )
    modified[0]["content"] += role_rule
    reference = [
        {"type": "text", "text": "IDENTITY REFERENCE ONLY (not the source frame):"},
        images[0],
    ]
    scene = [
        {"type": "text", "text": "SOURCE FRAME: report coordinates in this image only."},
        frame_label,
        images[1],
    ]
    modified[1]["content"] = (scene + reference if first_scene else reference + scene) + [
        {"type": "text", "text": target + role_rule}
    ]
    return modified


def normalize_integer_points(payload):
    """Fixed declared units, never inferred from magnitude, frame size, or labels."""
    result = extract_json_object(payload)
    if set(result) != {"present", "x", "y", "note"} or type(result["present"]) is not bool:
        raise ValueError("Invalid integer-coordinate response schema.")
    if not isinstance(result["note"], str):
        raise ValueError("A localization note must be text.")
    for axis in ("x", "y"):
        value = result[axis]
        if result["present"]:
            if type(value) is not int or not 0 <= value <= 1000:
                raise ValueError("Visible coordinates must be integers in 0..1000.")
            result[axis] = value / 1000
        elif value is not None:
            raise ValueError("Absent coordinates must be null.")
    return json.dumps(result)


def score_frames(frame_reports, evidence, labels, cutoff):
    expected = {entry["id"]: entry["at_ms"] for entry in evidence}
    actual = {entry["id"]: entry for entry in frame_reports}
    if frame_reports and (len(actual) != len(frame_reports) or set(actual) != set(expected)):
        raise ValueError("The review's frame inventory does not match its input.")
    rows = []
    for frame_id, at_ms in expected.items():
        if at_ms > cutoff:
            raise ValueError("A review frame crossed the cutoff.")
        index = round(at_ms * labels["assigned_fps"] / 1000)
        if index < 0 or index >= len(labels["labels"]):
            raise ValueError("Frame timestamp has no annotation.")
        truth = labels["labels"][index]
        if abs(truth["at_ms"] - at_ms) > 1:
            raise ValueError("Frame timestamp does not align with annotations.")
        prediction = actual.get(frame_id)
        if prediction and prediction["at_ms"] != at_ms:
            raise ValueError("A review changed a frame timestamp.")
        present = prediction["present"] if prediction else None
        x, y = (prediction.get("x"), prediction.get("y")) if prediction else (None, None)
        box = truth["box"]
        inside = False
        error = None
        if present and truth["visible"] and box and x is not None and y is not None:
            inside = box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]
            diagonal = math.hypot(box["x2"] - box["x1"], box["y2"] - box["y1"])
            error = (
                math.hypot(x - (box["x1"] + box["x2"]) / 2, y - (box["y1"] + box["y2"]) / 2)
                / diagonal
            )
        rows.append(
            {
                "at_ms": at_ms,
                "frame": truth["frame"],
                "truth_visible": truth["visible"],
                "truth_box": box,
                "truth_zone": truth["zone"],
                "available": prediction is not None,
                "predicted_visible": present,
                "x": x,
                "y": y,
                "predicted_zone": prediction.get("zone") if prediction else None,
                "point_inside_target_box": inside,
                "center_error_box_diagonals": error,
                "correct_presence": prediction is not None and present == truth["visible"],
            }
        )
    return rows


def summarize(rows):
    visible = sum(r["truth_visible"] for r in rows)
    positive = sum(r["predicted_visible"] is True for r in rows)
    correct = sum(r["point_inside_target_box"] for r in rows)
    return {
        "frame_requests": len(rows),
        "available_predictions": sum(r["available"] for r in rows),
        "truth_visible": visible,
        "truth_absent": len(rows) - visible,
        "positive_predictions": positive,
        "correct_presence": sum(r["correct_presence"] for r in rows),
        "correct_point_in_box": correct,
        "point_recall": correct / visible if visible else None,
        "point_precision": correct / positive if positive else None,
        "false_visible_when_absent": sum(
            r["predicted_visible"] is True and not r["truth_visible"] for r in rows
        ),
        "outside_target_box_when_visible": sum(
            r["predicted_visible"] is True
            and r["truth_visible"]
            and not r["point_inside_target_box"]
            for r in rows
        ),
    }


def cross_recording_absence(source_labels: dict, reference_labels: dict) -> dict:
    """Treat another recording's registered target as an identity-negative proxy.

    This is not a claim that the source has no object of the same category. Separate
    recordings do not prove different physical objects; inspect the scenes and report
    that limitation. Both recordings and their annotations remain unedited.
    """
    if source_labels["video_sha256"] == reference_labels["video_sha256"]:
        raise ValueError("A cross-recording negative needs two different source videos.")
    if source_labels["category"] != reference_labels["category"]:
        raise ValueError("Cross-recording comparisons must use the same object category.")
    labels = copy.deepcopy(source_labels)
    for row in labels["labels"]:
        row.update({"visible": False, "box": None, "zone": None})
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-edge",
        type=int,
        choices=[448, 640, 768, 896],
        help="Research override; does not change application configuration bounds.",
    )
    parser.add_argument(
        "--focus",
        choices=["off", "refine", "confirm"],
        help="Zoomed second-look arm. Overrides the environment so each arm is explicit.",
    )
    parser.add_argument(
        "--variant",
        choices=["deployed", "explicit_roles", "scene_first", "larger_frame", "integer_points"],
        default="deployed",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Research override of the per-frame generation budget, for models that reason "
        "before answering. The application keeps its own budget.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Preserve first reports: choose a new output path.")
    protocol_raw = args.protocol.read_bytes()
    protocol = json.loads(protocol_raw)
    provider = Providers(Settings())
    if args.variant == "larger_frame":
        provider.settings.cosmos_max_edge = 896
    if args.max_edge is not None:
        provider.settings.cosmos_max_edge = args.max_edge
    if args.focus is not None:
        provider.settings.cosmos_focus = args.focus
    if not provider.settings.cosmos_available:
        raise SystemExit("Configure the existing Cosmos service first.")
    body, trace = SkillRegistry().load("review-visual-evidence")
    source_root = Path(__file__).resolve().parents[2] / "src" / "agentx"
    report = {
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_root.rglob("*.py"))
        },
        "platform": platform.platform(),
        "model": provider.settings.cosmos_model,
        "variant": args.variant,
        "focus_pass": provider.settings.cosmos_focus,
        "focus_window": provider.settings.cosmos_focus_window,
        "max_edge": provider.settings.cosmos_max_edge,
        "reference_grounding": provider.settings.cosmos_reference_grounding,
        "reference_max_edge": provider.settings.cosmos_reference_max_edge,
        "max_new_tokens_override": args.max_new_tokens,
        "skill": trace.model_dump(),
        "windows": [],
        "scope": "Public-video frame localization against external labels. Cross-recording absence is an identity-negative proxy, not proof of distinct physical objects. Point-in-box is not IoU, broad accuracy or a user study. Repeated source frames remain correlated requests.",
    }
    calls, lock = [], threading.Lock()
    generate = provider._generate
    current = threading.local()
    # Token usage per request, so reasoning models' extra generation is visible next to latency.
    provider.usage_sink = lambda model, usage: setattr(current, "usage", usage)

    def record(messages, **kwargs):
        messages = variant_messages(messages, args.variant)
        if args.max_new_tokens is not None:
            kwargs["max_tokens"] = args.max_new_tokens
        start = time.monotonic()
        call = {
            "messages_sha256": hashlib.sha256(
                json.dumps(messages, sort_keys=True).encode()
            ).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(messages[0]["content"].encode()).hexdigest(),
        }
        try:
            current.usage = None
            value = generate(messages, **kwargs)
            call["response"] = value
            call["usage"] = current.usage
            if args.variant == "integer_points":
                try:
                    value = normalize_integer_points(value)
                    call["normalized_response"] = value
                except ValueError as exc:
                    call["normalization_error"] = str(exc)
                    # Let the unchanged production validator perform its usual retry.
                    # Never let decimal outputs accidentally pass as 0..1 coordinates.
                    value = '{"invalid_integer_coordinate_contract":true}'
            return value
        except Exception as exc:
            call["error_type"] = type(exc).__name__
            raise
        finally:
            call["seconds"] = round(time.monotonic() - start, 4)
            with lock:
                calls.append(call)

    provider._generate = record
    for case in protocol["windows"]:
        directory = args.dataset / case["sequence"]
        raw = (directory / "labels.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != protocol["inputs"][case["sequence"]]["labels_sha256"]:
            raise ValueError("Labels changed after the protocol freeze.")
        source_labels = json.loads(raw)
        source = directory / f"{case['sequence']}.mp4"
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != source_labels["video_sha256"]:
                raise ValueError("Video no longer matches the annotations.")
        reference_sequence = case.get("reference_sequence", case["sequence"])
        if reference_sequence != case["sequence"]:
            if case.get("truth") != "different_recording_absent" or not case["reference"]:
                raise ValueError("Cross-recording negatives need an explicit truth and crop.")
            ref_dir = args.dataset / reference_sequence
            ref_raw = (ref_dir / "labels.json").read_bytes()
            if (
                hashlib.sha256(ref_raw).hexdigest()
                != protocol["inputs"][reference_sequence]["labels_sha256"]
            ):
                raise ValueError("Reference labels changed after the protocol freeze.")
            reference_labels = json.loads(ref_raw)
            reference_source = ref_dir / f"{reference_sequence}.mp4"
            with reference_source.open("rb") as stream:
                if (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    != reference_labels["video_sha256"]
                ):
                    raise ValueError("Reference video no longer matches its annotations.")
            labels = cross_recording_absence(source_labels, reference_labels)
        else:
            if case.get("truth") == "different_recording_absent":
                raise ValueError("The reference and source recording must differ.")
            reference_labels = source_labels
            reference_source = source
            labels = source_labels
        _, first = read_frame(reference_source, 0)
        box = reference_labels["labels"][0]["box"]
        h, w = first.shape[:2]
        crop = jpeg(
            first[int(box["y1"] * h) : int(box["y2"] * h), int(box["x1"] * w) : int(box["x2"] * w)]
        )
        evidence, _ = provider._sample(source, case["cutoff_ms"])
        calls.clear()
        start = time.monotonic()
        result, error_type = None, None
        try:
            result = provider.review(
                source,
                case["cutoff_ms"],
                "Where is the registered target visible?",
                [body],
                target=f"Registered {labels['category']}",
                regions=default_regions(),
                reference=crop if case["reference"] else None,
            )
            if args.variant != "deployed":
                result["provenance"]["base_prompt_sha256"] = result["provenance"]["prompt_sha256"]
                result["provenance"]["prompt_sha256"] = calls[0]["system_prompt_sha256"]
                result["provenance"]["research_variant"] = args.variant
                if args.variant == "integer_points":
                    result["provenance"]["model_coordinate_units"] = "integer_0_1000_per_axis"
                    result["provenance"]["coordinate_conversion"] = "strict_integer_divide_1000"
            rows = score_frames(result["frame_reports"], evidence, labels, case["cutoff_ms"])
        except Exception as exc:
            error_type = type(exc).__name__
            rows = score_frames([], evidence, labels, case["cutoff_ms"])
        window = {
            **case,
            "status": "failed" if error_type else "complete",
            "error_type": error_type,
            "elapsed_seconds": round(time.monotonic() - start, 4),
            "reference_sha256": hashlib.sha256(crop).hexdigest() if case["reference"] else None,
            "result": result,
            "calls": list(calls),
            "rows": rows,
            "summary": summarize(rows),
        }
        report["windows"].append(window)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: window[k]
                    for k in (
                        "sequence",
                        "cutoff_ms",
                        "reference",
                        "status",
                        "elapsed_seconds",
                        "summary",
                    )
                }
            ),
            flush=True,
        )
    report["status"] = "complete"
    report["failed_windows"] = sum(w["status"] != "complete" for w in report["windows"])
    report["summary"] = {
        mode: summarize(
            [r for w in report["windows"] if w["reference"] == use_reference for r in w["rows"]]
        )
        for mode, use_reference in [("with_reference", True), ("without_reference", False)]
    }
    report["unique_source_frames"] = len(
        {(w["sequence"], r["at_ms"]) for w in report["windows"] for r in w["rows"]}
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return int(bool(report["failed_windows"]))


if __name__ == "__main__":
    raise SystemExit(main())
