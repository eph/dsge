"""
MkDocs Macros: inject package version into docs.

Usage in Markdown:
  - Current version: {{ version }}
"""

from __future__ import annotations

from pathlib import Path


def _resolve_version() -> str:
    # Prefer installed metadata when available
    try:
        try:
            from importlib.metadata import version as _pkg_version  # type: ignore
        except Exception:  # pragma: no cover
            from importlib_metadata import version as _pkg_version  # type: ignore
        return _pkg_version("dsge")
    except Exception:
        pass

    # Fallback to setuptools-scm from git (source checkout)
    try:
        from setuptools_scm import get_version as _scm_version  # type: ignore
        # docs/macros.py -> project root is one parent up from this file
        project_root = Path(__file__).resolve().parent.parent
        return _scm_version(root=str(project_root))
    except Exception:
        return "0.0.0"


def define_env(env):
    """Register variables and filters for mkdocs-macros-plugin."""
    ver = _resolve_version()
    env.variables["version"] = ver
    # Also expose to theme templates via config.extra
    try:
        extra = env.conf.get("extra", {})
        extra["version"] = ver
        env.conf["extra"] = extra
    except Exception:
        pass
