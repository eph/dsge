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

from contextlib import contextmanager
from pathlib import Path
import tempfile

try:  # Python 3.9+
    from importlib.resources import files, as_file  # type: ignore
except Exception:  # pragma: no cover - fallback for Python 3.8
    from importlib_resources import files, as_file  # type: ignore


def _resource(rel: str):
    rel = rel.lstrip("/").replace("\\", "/")
    return files("dsge").joinpath(*rel.split("/"))


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

# Persistent extraction cache for resources that need a stable filesystem path
_CACHE_BASE = Path(tempfile.gettempdir()) / "dsge_resources_cache"

def resource_file_path(rel: str) -> Path:
    """
    Return a persistent filesystem path to a packaged resource.

    Copies the resource into a temp cache directory if it isn't already on the
    filesystem. Suitable for configs that need a long-lived path string.
    """
    rel_norm = rel.lstrip("/").replace("\\", "/")
    dest = _CACHE_BASE / rel_norm
    if dest.exists():
        return dest
    res = _resource(rel_norm)
    data = res.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    return dest

__all__.append("resource_file_path")
