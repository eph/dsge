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

# Validation utilities
from .validation import validate_model_consistency

# Version information
__version__ = '0.0.4'  # Update this with your actual version

