#!/usr/bin/env python3
"""
Logging configuration for the DSGE package.

This module sets up a standardized logging configuration to be used
throughout the package, making log output consistent and configurable.
"""

import logging
import sys
from typing import Any, Dict, Optional

# Default logging format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Library default: be silent unless the user opts in.
#
# We intentionally do *not* configure handlers on import, because that would
# force log output on downstream users. Instead, we attach a NullHandler and
# set a conservative default level so `logger.info(...)` calls are hidden unless
# logging is explicitly enabled by the application (or via `configure_logging`).
_pkg_logger = logging.getLogger("dsge")
_pkg_logger.addHandler(logging.NullHandler())
_pkg_logger.setLevel(logging.WARNING)



def configure_logging(
    level: int = logging.INFO,
    format_str: Optional[str] = None,
    handlers: Optional[Dict[str, Any]] = None
) -> None:
    """
    Configure the logging system for the DSGE package.
    
    Args:
        level: The logging level (default: INFO)
        format_str: The log format string (default: timestamp, logger name, level, message)
        handlers: Dictionary of handlers to configure
    """
    format_str = format_str or DEFAULT_FORMAT
    formatter = logging.Formatter(format_str)
    
    # Clear any existing handlers
    root_logger = logging.getLogger('dsge')
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set the logging level
    root_logger.setLevel(level)
    
    # Add a default console handler if none provided
    if not handlers:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    else:
        # Configure provided handlers
        for handler in handlers.values():
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
    
    # Prevent propagation to the root logger
    root_logger.propagate = False
    
    # Log configuration completed
    root_logger.debug("Logging configured for DSGE package")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module within the DSGE package.
    
    Args:
        name: The name of the module or component
        
    Returns:
        A configured logger instance
    """
    return logging.getLogger(f"dsge.{name}")


__all__ = ["configure_logging", "get_logger", "DEFAULT_FORMAT"]
