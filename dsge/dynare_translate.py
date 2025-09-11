from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def _apply_correlations(diag_cov: Dict[str, str], corrs: List[Tuple[str, str, str]]) -> Dict[str, str]:
    cov = dict(diag_cov)
    for n1, n2, rho in corrs:
        v1 = diag_cov.get(n1)
        v2 = diag_cov.get(n2)
        if v1 is None or v2 is None:
            # Skip if we don't have both variances defined
            continue
        cov[f"{n1},{n2}"] = f"({rho})*sqrt(({v1}))*sqrt(({v2}))"
    return cov


def to_yaml_like(parsed: Dict[str, Any], name: str = "dynare_model") -> Dict[str, Any]:
    """
    Convert parsed Dynare dict to the YAML-like dict expected by DSGE.read.
    """
    variables = parsed.get("variables", [])
    shocks = parsed.get("shocks", [])
    parameters = parsed.get("parameters", [])
    equations = parsed.get("equations", [])
    param_values = parsed.get("param_values", {})
    covariance = parsed.get("covariance", {})
    correlations = parsed.get("correlations", [])
    covariance = _apply_correlations(covariance, correlations)
    initval = parsed.get("initval", {})
    endval = parsed.get("endval", {})
    varexo_det = parsed.get("varexo_det", [])
    predetermined = parsed.get("predetermined", [])
    observables = parsed.get("observables", [])

    external: Dict[str, Any] = {}
    if initval:
        external["initval"] = initval
    if endval:
        external["endval"] = endval
    if varexo_det:
        external["varexo_det"] = varexo_det
    if predetermined:
        external["predetermined_variables"] = predetermined

    model_yaml: Dict[str, Any] = {
        "declarations": {
            "name": name,
            "type": "dsge",
            "variables": variables,
            "parameters": parameters,
            "shocks": shocks,
            # Keep extras under external so schema accepts it
            **({"observables": observables} if observables else {}),
            **({"external": external} if external else {}),
        },
        "equations": {
            "model": [eq if ("=" in eq) else f"{eq} = 0" for eq in equations],
        },
        "calibration": {
            "parameters": param_values,
            "covariance": covariance,
        },
        # estimation is optional; leave empty
    }

    return model_yaml
