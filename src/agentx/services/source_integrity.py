"""Bounded, change-aware verification of the original recording's import hash."""

import hashlib
import stat
from collections import OrderedDict
from pathlib import Path
from threading import Lock


class SourceIntegrityError(ValueError):
    """A source verification failure with a public message that omits storage details."""


class SourceIntegrity:
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._cache: OrderedDict[tuple, str | None] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def _signature(path: Path) -> tuple:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise FileNotFoundError("Original evidence is missing.")
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def problem(self, path: Path, expected_sha256: str) -> str | None:
        """None means verified; a changed inode/size/mtime/ctime always invalidates the cache.

        This detects ordinary file damage/replacement, not a privileged adversary who
        controls both files and filesystem metadata. Exports independently hash again.
        """
        try:
            with self._lock:
                before = self._signature(path)
                key = (str(path), expected_sha256, *before)
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if before != self._signature(path):
                    return "evidence_changed"
                problem = None if digest == expected_sha256 else "evidence_changed"
                self._cache[key] = problem
                while len(self._cache) > self.capacity:
                    self._cache.popitem(last=False)
                return problem
        except FileNotFoundError:
            return "evidence_missing"
        except OSError:
            return "evidence_unavailable"
