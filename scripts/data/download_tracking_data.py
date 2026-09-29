"""Download pinned LaSOT research archives with parallel Hugging Face/Xet transfers.

The source tree contains no third-party footage. Downloaded archives remain local
research artifacts. Dataset rights are separate from the toolkit's code license.
Use --endpoint to benchmark a reachable mirror; keep the revision and hashes fixed.
"""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "l-lt/LaSOT"
REVISION = "a97464600e8c1ab91eb91a47332a20732830585e"
SHA256 = {
    "book": "15b5ab6c68f5ab6d76903114a4daf97edf711bad86d553fc6b7285b0ffdd3bfc",
    "bottle": "86d894a6e8c2e74f00dee11f3830321fcb34ce377b1dcb729508ab6f90cfedf2",
    "cup": "5a6708bd99a4753c935a07be1e923eef0a343a25e3caa539cdfdfde8b5cc96f8",
    "mouse": "c4fa66cedb87109c92b3b770e5133b2ab00bbc82d14af8c9e7f0b5e3176c7eac",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=sorted(SHA256), default=["cup", "book"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/datasets/lasot-raw"))
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    # Set before importing the Hub/Xet clients so their configuration sees it.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import hf_hub_download

    args.output.mkdir(parents=True, exist_ok=True)

    def download(category):
        started = time.monotonic()
        path = Path(
            hf_hub_download(
                REPO,
                f"{category}.zip",
                repo_type="dataset",
                revision=REVISION,
                local_dir=args.output,
                endpoint=args.endpoint,
            )
        )
        elapsed = time.monotonic() - started
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if sha != SHA256[category]:
            raise ValueError(f"Archive checksum mismatch: {path.name}")
        row = {
            "category": category,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha,
            "download_seconds": round(elapsed, 3),
        }
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(download, dict.fromkeys(args.categories)))
    (args.output / "download-manifest.json").write_text(
        json.dumps(
            {
                "repo": REPO,
                "revision": REVISION,
                "endpoint": args.endpoint,
                "purpose": "Public tracking research stress test; not independent desk footage or user trials.",
                "archives": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
