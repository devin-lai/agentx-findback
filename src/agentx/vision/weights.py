"""Process-wide cache of read-only inference weights.

Loading SAM 2.1, DINOv2 and RT-DETR from disk costs seconds for every memory run; for a live
capture that is seconds of camera the memory has to catch up on before it can answer. The cached
models are in eval mode and only run under inference mode. Per-video state never lives on them:
SAM 2.1 keeps it in its own session object, and the other two are single-image models.
"""

import hashlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

_cache: dict[tuple, Any] = {}
_lock = threading.Lock()


def cached[T](key: tuple, load: Callable[[], T]) -> T:
    """The value stored under `key`, loading it once. Loads are serialized."""
    with _lock:
        if key not in _cache:
            _cache[key] = load()
        return _cache[key]


def file_sha256(path: Path) -> str:
    """A checkpoint's hash, recomputed only when its size or modification time changes."""
    info = path.stat()

    def digest() -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    return cached(("sha256", str(path.resolve()), info.st_size, info.st_mtime_ns), digest)


def clear() -> None:
    with _lock:
        _cache.clear()
