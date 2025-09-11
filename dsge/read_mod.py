from __future__ import annotations

from pathlib import Path

from .dynare_parser import parse_mod_file
from .dynare_translate import to_yaml_like
from .DSGE import DSGE


def read_mod(path: str | Path):
    """
    Read a Dynare .mod file and return a DSGE model object.

    Minimal support: var, varexo, parameters (+ assignments), model/end, shocks/end (var or stderr), initval/end.
    Advanced Dynare features (macros, loops, steady_state_model) are not supported in v1.
    """
    p = Path(path)
    parsed = parse_mod_file(p)
    yaml_like = to_yaml_like(parsed, name=p.stem)
    # Build model using existing DSGE flow
    model = DSGE.read(yaml_like)
    return model

