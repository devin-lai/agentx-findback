"""Re-export saved development answers without re-indexing footage or invoking a model."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import httpx

from agentx.config import Settings
from agentx.services.bundle_verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    args = parser.parse_args()
    source = args.receipt.read_bytes()
    receipt = json.loads(source)
    if receipt.get("status") != "complete":
        raise SystemExit("Use a completed live receipt.")
    if args.output.exists():
        raise SystemExit("Choose a new output directory to preserve earlier exports.")
    args.output.mkdir(parents=True)
    settings = Settings()
    headers = {"Authorization": f"Bearer {settings.api_token}"} if settings.api_token else {}
    exported = []
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=120) as client:
        for number, item in enumerate(receipt["answers"], start=1):
            answer = item["answer"]
            response = client.get(f"/api/v1/questions/{answer['id']}/bundle")
            response.raise_for_status()
            archive_path = args.output / f"answer-{number}.zip"
            archive_path.write_bytes(response.content)
            folder = args.output / f"answer-{number}"
            with ZipFile(io.BytesIO(response.content)) as archive:
                archive.extractall(folder)
            result = verify(folder)
            exported.append(
                {
                    "name": item["name"],
                    "query_id": answer["id"],
                    "archive": archive_path.name,
                    "sha256": hashlib.sha256(response.content).hexdigest(),
                    "verification": result,
                }
            )
            print(json.dumps(exported[-1]), flush=True)
    (args.output / "exports.json").write_text(
        json.dumps(
            {
                "receipt_sha256": hashlib.sha256(source).hexdigest(),
                "exports": exported,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
