"""
DSGE: A Python package for Dynamic Stochastic General Equilibrium models.

This package provides tools for defining, solving, and estimating DSGE models
using various approaches including linear rational expectations models and
full information home production (FHP) models.
"""

# Configure logging first
from .logging_config import configure_logging, get_logger

# Core model classes
from .DSGE import DSGE
from .FHPRepAgent import FHPRepAgent
from .SIDSGE import SIDSGE

# YAML parsing and utilities
from .parse_yaml import read_yaml
from .read_mod import read_mod

# Validation utilities
from .validation import validate_model_consistency

# Code generation utilities
from .translate import translate_fortran, make_fortran_model
from .translate_cpp import translate_cpp

# IRF-based counterfactual utilities (experimental)
from .irfoc import IRFOC, IRFBasedCounterfactual

# Version information via installed metadata or SCM fallback
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFound
except Exception:  # pragma: no cover
    try:
        from importlib_metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFound  # type: ignore
    except Exception:  # final fallback
        _pkg_version = None  # type: ignore
        class _PkgNotFound(Exception):
            pass

try:
    __version__ = _pkg_version("dsge") if _pkg_version else "0.0.0"
except _PkgNotFound:
    try:
        # Attempt to derive from git via setuptools-scm when not installed
        from setuptools_scm import get_version as _scm_version  # type: ignore
        __version__ = _scm_version(root="..", relative_to=__file__)
    except Exception:
        __version__ = "0.0.0"
