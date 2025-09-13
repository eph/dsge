"""
Occasionally Binding Constraints (OBC) support (skeleton).

Provides an OccBin-style wrapper model that holds per-regime linear DSGE models
and constraint metadata, plus placeholder IRF/simulation methods.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np

from .DSGE import DSGE


class OccBinModel:
    """Wrapper for an OBC model with two regimes and a single constraint.

    Attributes
    - regimes: dict[str, DSGE] parsed from YAML 'regimes'
    - constraint: dict with keys: name, when (str), binding_regime, normal_regime, horizon (opt)
    - observables, parameters, shocks: shared across regimes
    - context: parsing context (if needed later)
    """

    def __init__(self, *, regimes: Dict[str, DSGE], constraint: Dict[str, Any], yaml: Dict[str, Any]):
        self.regimes = regimes
        self.constraint = constraint
        self.yaml = yaml

        # Borrow shared attributes from one regime (we require consistency)
        any_regime = next(iter(regimes.values()))
        self.parameters = any_regime.parameters
        self.shocks = any_regime.shocks
        self.variables = any_regime.variables
        self.observables = any_regime["observables"]

    def p0(self):
        return list(map(lambda x: self.yaml["calibration"]["parameters"][str(x)], self.parameters))

    def compile_regime(self, name: str):
        return self.regimes[name].compile_model()

    # Placeholder solver API — use normal regime IRF for now
    def irf_obc(self, p0, shock_name: Optional[str] = None, h: int = 20):
        normal_name = self.constraint.get("normal_regime")
        if normal_name is None:
            normal_name = list(self.regimes.keys())[0]
        model = self.compile_regime(normal_name)
        irfs = model.impulse_response(p0, h)
        regime_flags = np.zeros((h + 1,), dtype=bool)  # no binding in placeholder
        return {"irf": irfs, "binding": regime_flags}


def read_obc(model_yaml: Dict[str, Any]) -> OccBinModel:
    """Construct an OccBinModel from YAML with 'regimes' and 'constraints'.

    Expects one constraint for now. Regime models are built by reusing DSGE.read
    with the same declarations/calibration but regime-specific equations.
    """
    dec = model_yaml["declarations"]
    cal = model_yaml["calibration"]

    regimes_yaml: Dict[str, Any] = model_yaml.get("regimes", {})
    constraints = model_yaml.get("constraints", [])
    if not regimes_yaml or not constraints:
        raise ValueError("OBC YAML must include 'regimes' and a non-empty 'constraints' list")

    # Build DSGE models per regime by reconstructing a minimal YAML per regime
    regimes: Dict[str, DSGE] = {}
    for name, reg in regimes_yaml.items():
        eqs = reg.get("equations") or reg
        if isinstance(eqs, dict):
            eq_block = {"equations": eqs}
        else:
            eq_block = {"equations": {"model": eqs, "observables": model_yaml.get("equations", {}).get("observables", {})}}

        reg_yaml = {
            "declarations": dec,
            "calibration": cal,
            **eq_block,
        }
        regimes[name] = DSGE.read(reg_yaml)

    # For now, support single constraint
    constraint = constraints[0]
    return OccBinModel(regimes=regimes, constraint=constraint, yaml=model_yaml)

