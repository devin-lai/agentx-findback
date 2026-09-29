"""Fetch a pinned safetensors snapshot through the fastest reachable verified route.

The manifest supplies official SHA-256 values and a Hugging Face revision. Probe
ModelScope, Hugging Face and hf-mirror concurrently; fall back on failed routes.
No repository code is downloaded or executed. Hash mismatch never installs a file.
"""

import argparse
import hashlib
import json
import re
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def urls(repository, revision, name):
    query = urllib.parse.urlencode({"Revision": "master", "FilePath": name})
    return {
        "modelscope": f"https://modelscope.cn/api/v1/models/{repository}/repo?{query}",
        "huggingface": f"https://huggingface.co/{repository}/resolve/{revision}/{name}",
        "hf-mirror": f"https://hf-mirror.com/{repository}/resolve/{revision}/{name}",
    }


def validate_manifest(manifest):
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", manifest["repository"]):
        raise ValueError("Expected an owner/repository name.")
    if not re.fullmatch(r"[a-f0-9]{40}", manifest["revision"]):
        raise ValueError("A pinned commit revision is required.")
    for name, digest in manifest["files"].items():
        if not re.fullmatch(r"[\w.-]+\.(json|safetensors)", name):
            raise ValueError("Only plain JSON configuration and safetensors files are allowed.")
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Every file needs an official SHA-256 value.")
    if "model.safetensors" not in manifest["files"]:
        raise ValueError("A safetensors checkpoint is required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    routes = urls(manifest["repository"], manifest["revision"], "model.safetensors")

    def probe(item):
        route, url = item
        start = time.monotonic()
        try:
            request = urllib.request.Request(url, headers={"Range": "bytes=0-65535"})
            with urllib.request.urlopen(request, timeout=8) as response:
                data = response.read(65536)
            if len(data) != 65536:
                raise ValueError("Incomplete model probe")
            return dict(route=route, seconds=time.monotonic() - start, reachable=True)
        except Exception as exc:
            return dict(
                route=route,
                seconds=time.monotonic() - start,
                reachable=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    with ThreadPoolExecutor(max_workers=3) as pool:
        probes = list(pool.map(probe, routes.items()))
    print(json.dumps(dict(probes=probes)), flush=True)
    chosen = [p["route"] for p in sorted(probes, key=lambda p: p["seconds"]) if p["reachable"]]
    # A failed probe can be transient. Try those routes last if necessary.
    chosen += [p["route"] for p in probes if not p["reachable"]]

    def download(item):
        name, expected = item
        target = args.output / name
        if target.exists():
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == expected:
                    return dict(
                        file=name, cached=True, bytes=target.stat().st_size, sha256=expected
                    )
        failures = []
        for route in chosen:
            part = None
            start = time.monotonic()
            try:
                with tempfile.NamedTemporaryFile(
                    dir=args.output, prefix=name, suffix=".part", delete=False
                ) as sink:
                    part = Path(sink.name)
                    digest = hashlib.sha256()
                    with urllib.request.urlopen(
                        urls(manifest["repository"], manifest["revision"], name)[route], timeout=30
                    ) as source:
                        while chunk := source.read(2 * 1024 * 1024):
                            sink.write(chunk)
                            digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError("Official checkpoint checksum mismatch")
                part.replace(target)
                row = dict(
                    file=name,
                    bytes=target.stat().st_size,
                    sha256=expected,
                    route=route,
                    seconds=time.monotonic() - start,
                    failed_attempts=failures,
                )
                print(json.dumps(row), flush=True)
                return row
            except Exception as exc:
                failures.append(dict(route=route, error=f"{type(exc).__name__}: {exc}"))
            finally:
                if part is not None:
                    part.unlink(missing_ok=True)
        raise ValueError(f"No verified route for {name}: {failures}")

    with ThreadPoolExecutor(max_workers=5) as pool:
        files = list(pool.map(download, manifest["files"].items()))
    report = dict(
        manifest=manifest, probes=probes, files=files, elapsed_seconds=time.monotonic() - started
    )
    (args.output / "agentx-download.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
