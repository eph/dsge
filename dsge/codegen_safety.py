"""Validation at the boundary between model data and generated source code."""

import math
import re
from numbers import Integral

import sympy


def code_identifier(value: str, *, language: str) -> str:
    """Return an ASCII identifier accepted by the supported source languages."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid {language} identifier: {value!r}")
    if len(value) > 63:
        raise ValueError(f"{language} identifiers must have at most 63 characters: {value!r}")
    return value


def single_line_text(value: str, *, field: str) -> str:
    if not isinstance(value, str) or any(not c.isprintable() for c in value):
        raise ValueError(f"{field} must be text without control characters")
    return value


def fortran_string(value: str) -> str:
    """Encode a character expression, including quotes and compiler escape modes."""
    value = single_line_text(value, field="Fortran string")
    # Backslashes are emitted as ACHAR so -fbackslash cannot reinterpret them.
    parts = []
    for i, part in enumerate(value.split("\\")):
        if i:
            parts.append("achar(92)")
        for start in range(0, max(1, len(part)), 40):
            parts.append("'" + part[start:start + 40].replace("'", "''") + "'")
    return " // &\n      ".join(parts)


def code_integer(value, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, sympy.Integer)):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def fortran_real(value, *, allow_nan: bool = False) -> str:
    """Serialize a real scalar without trusting its textual representation."""
    value = float(value)
    if allow_nan and math.isnan(value):
        return "ieee_value(0.0_wp, ieee_quiet_nan)"
    if not math.isfinite(value):
        raise ValueError("Fortran numeric values must be finite")
    return f"{value}_wp"


def validate_fortran_expression(expr) -> None:
    """Check model-controlled names before passing an AST to SymPy's printer."""
    for symbol in expr.atoms(sympy.Symbol):
        code_identifier(symbol.name, language="Fortran")
    for function in expr.atoms(sympy.Function):
        code_identifier(function.func.__name__, language="Fortran")
