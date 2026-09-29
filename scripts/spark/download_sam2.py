"""Fetch SAM 2.1 Small from reachable ModelScope, verifying official HF file hashes.

No model code is executed and no environment packages are installed. The files
are the Transformers safetensors export of facebook/sam2.1-hiera-small at
ee5bba1d82bb8749febdf90f45e84b687142ba03, licensed Apache-2.0 by Meta.
"""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FILES = {
    "model.safetensors": "0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60",
    "config.json": "97ff9f65b76d107acda4247885f0a5555d0048850ae3c5f97183df289aaecde9",
    "processor_config.json": "f8a68e865cfad115c1c2763f3d93eca7b1c622da06da2a9273eb437fb2389b6d",
    "video_preprocessor_config.json": "9fccfe5f464ec38c2f236d0e6a68e95511c80c22132fc2fa4b9f7b65f24fad95",
    "preprocessor_config.json": "6ebf229ee259368ce4a8d4f2fe893a72b053023710853e257253939e601f583d",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path.home() / "models" / "sam2.1-hiera-small"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def download(item):
        name, expected = item
        target = args.output / name
        if target.exists():
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == expected:
                    return {"file": name, "cached": True, "bytes": target.stat().st_size}
        query = urllib.parse.urlencode({"Revision": "master", "FilePath": name})
        url = "https://modelscope.cn/api/v1/models/facebook/sam2.1-hiera-small/repo?" + query
        digest = hashlib.sha256()
        part = target.with_suffix(target.suffix + ".part")
        t0 = time.monotonic()
        with urllib.request.urlopen(url, timeout=60) as source, part.open("wb") as sink:
            while chunk := source.read(2 * 1024 * 1024):
                digest.update(chunk)
                sink.write(chunk)
        if digest.hexdigest() != expected:
            part.unlink()
            raise ValueError(f"Official checkpoint hash mismatch for {name}; no file installed.")
        part.replace(target)
        row = {
            "file": name,
            "bytes": target.stat().st_size,
            "seconds": round(time.monotonic() - t0, 3),
            "sha256": expected,
        }
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(download, FILES.items()))
    report = {
        "source": "ModelScope facebook/sam2.1-hiera-small; hashes verified against official Hugging Face revision",
        "official_revision": "ee5bba1d82bb8749febdf90f45e84b687142ba03",
        "elapsed_seconds": time.monotonic() - started,
        "files": rows,
    }
    (args.output / "agentx-download.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
