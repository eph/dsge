"""
Code generation helpers for building Fortran SMC scaffolding.

This integrates with the external `fortress` package. Because `fortress` lives
on GitHub, it might not be installed in all environments (e.g., CI without
network). To keep `dsge` importable and tests reliable, we import fortress
optionally and skip features that require it when unavailable.
"""

from typing import Any, Dict

DEFAULT_MODEL_DIR = '__fortress_tmp'

try:
    # Optional dependency: available via the `codegen` extra, or if installed directly.
    from fortress import make_smc as _make_smc  # type: ignore
except Exception:  # pragma: no cover - environment-specific
    _make_smc = None  # type: ignore


def create_fortran_smc(
    model_file: str,
    output_directory: str = DEFAULT_MODEL_DIR,
    other_files: Dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Create a Fortran SMC project using Fortress.

    Raises unittest.SkipTest if Fortress is not installed so tests are skipped
    rather than fail in environments without network access.
    """
    if _make_smc is None:
        import unittest

        raise unittest.SkipTest(
            "fortress is not installed. Install optional extra: dsge[codegen]"
        )

    if other_files is None:
        other_files = {}
    # Pass all arguments as keyword arguments to match make_smc's keyword-only API
    return _make_smc(model_file, output_directory=output_directory, other_files=other_files, **kwargs)
