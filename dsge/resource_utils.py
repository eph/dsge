"""
Utilities for accessing package resources in a zip-safe way.

- resource_path: context manager yielding a real filesystem Path
  for a packaged resource (uses importlib.resources.as_file under the hood).
- open_text/open_binary: convenience readers for small files.

Example:
    from dsge.resource_utils import resource_path
    with resource_path('examples/ar1/ar1.yaml') as p:
        model = read_yaml(str(p))
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
import atexit
import threading

try:  # Python 3.9+
    from importlib.resources import files, as_file  # type: ignore
except Exception:  # pragma: no cover - fallback for Python 3.8
    from importlib_resources import files, as_file  # type: ignore


def _resource(rel: str):
    path = PurePosixPath(rel)
    if (not rel or path.is_absolute() or PureWindowsPath(rel).drive
            or ".." in path.parts or "\\" in rel):
        raise ValueError("Resource paths must be relative package paths without traversal")
    return files("dsge").joinpath(*path.parts)


@contextmanager
def resource_path(rel: str):
    """Yield a filesystem Path for a resource inside the dsge package."""
    res = _resource(rel)
    with as_file(res) as p:
        yield Path(p)


def open_text(rel: str, encoding: str = "utf-8"):
    """Open a text resource for reading."""
    return _resource(rel).open("r", encoding=encoding)


def open_binary(rel: str):
    """Open a binary resource for reading."""
    return _resource(rel).open("rb")


__all__ = ["resource_path", "open_text", "open_binary"]

# Keep importlib's exclusively-created temporary files alive for callers that
# need paths beyond one context manager. No shared, predictable cache is used.
_RESOURCE_STACK = ExitStack()
_RESOURCE_LOCK = threading.RLock()
atexit.register(_RESOURCE_STACK.close)


def resource_file_path(rel: str) -> Path:
    """Return a package resource path valid until this process exits.

    Filesystem packages use their original path. Zip resources are materialized
    by importlib.resources in a fresh private temporary file for this call.
    """
    res = _resource(rel)
    with _RESOURCE_LOCK:
        return Path(_RESOURCE_STACK.enter_context(as_file(res)))


__all__.append("resource_file_path")
