"""Load repository scripts by name, wherever they live under scripts/<area>/."""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def script_path(name: str) -> Path:
    matches = sorted(SCRIPTS.glob(f"*/{name}.py"))
    if len(matches) != 1:
        raise LookupError(f"Expected exactly one scripts/*/{name}.py, found {matches}")
    return matches[0]


def script_dir(area: str) -> str:
    return str(SCRIPTS / area)


def load_script(name: str):
    path = script_path(name)
    # Scripts import helpers that sit beside them, as they do when run from the command line.
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
