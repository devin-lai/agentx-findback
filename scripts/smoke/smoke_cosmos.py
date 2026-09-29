"""Execute the configured Cosmos model on a generated fixture; not an accuracy benchmark."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from agentx.agents.providers import Providers
from agentx.agents.skills import SkillRegistry
from agentx.config import Settings
from agentx.services.demo import create_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--without-review-skills", action="store_true")
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument(
        "--grounded",
        action="store_true",
        help="Per-frame localization of the fixture's red toolkit with its reference crop, "
        "ordered in code, instead of one free-form multi-image answer.",
    )
    parser.add_argument(
        "--cutoff-ms", type=int, default=7000, choices=range(0, 12001), metavar="0..12000"
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/cosmos-smoke.json"))
    args = parser.parse_args()
    settings = Settings()
    if not settings.cosmos_available:
        raise SystemExit("Configure the Cosmos HTTP service or a local checkpoint in .env first.")
    if args.require_cuda:
        if settings.cosmos_backend != "transformers":
            raise SystemExit(
                "This check can verify CUDA only for native local inference, not an HTTP server."
            )
        settings.cosmos_device = "cuda"
    body, trace = SkillRegistry().load("review-visual-evidence")
    with tempfile.TemporaryDirectory(prefix="agentx-cosmos-") as directory:
        source = Path(directory) / "fixture.mp4"
        objects = create_fixture(source)
        question = "Where was the red patterned square last visible? Does its disappearance prove that it entered a container?"
        target = regions = reference = None
        if args.grounded:
            from agentx.domain.contracts import default_regions
            from agentx.vision.video import jpeg, read_frame

            first = objects[0]
            _, frame = read_frame(source, 0)
            h, w = frame.shape[:2]
            b = first.box
            reference = jpeg(frame[int(b.y1 * h) : int(b.y2 * h), int(b.x1 * w) : int(b.x2 * w)])
            target, regions = first.name, default_regions()
        result = Providers(settings).review(
            source,
            args.cutoff_ms,
            question,
            [] if args.without_review_skills else [body],
            target=target,
            regions=regions,
            reference=reference,
        )
        report = {
            "scope": "Real model execution on synthetic imagery; schema validity is not answer accuracy.",
            "input": {
                "kind": "generated software fixture",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            "question": question,
            "cutoff_ms": args.cutoff_ms,
            "mode": result["mode"],
            "expected": "Red toolkit: left 0-3 s, right 3-6 s, occluded 6-8 s, center from 8 s.",
            "skills": [] if args.without_review_skills else [trace.model_dump()],
            "result": result,
        }
    if args.hash_weights and settings.cosmos_backend == "transformers":
        checkpoints = sorted(Path(settings.cosmos_model_path).glob("*.safetensors"))
        report["weight_sha256"] = {}
        for checkpoint in checkpoints:
            with checkpoint.open("rb") as stream:
                report["weight_sha256"][checkpoint.name] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "seconds": result["elapsed_seconds"],
                "provenance": result["provenance"],
                "accuracy": "requires independent human annotation",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
