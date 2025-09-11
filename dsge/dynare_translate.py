from __future__ import annotations

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
    # Currently not used by core YAML schema; kept for future mapping if needed
    # initval = parsed.get("initval", {})
    # endval = parsed.get("endval", {})
    # varexo_det = parsed.get("varexo_det", [])
    # predetermined = parsed.get("predetermined", [])
    observables = parsed.get("observables", [])

    model_yaml: Dict[str, Any] = {
        "declarations": {
            "name": name,
            "type": "dsge",
            "variables": variables,
            "parameters": parameters,
            "shocks": shocks,
            **({"observables": observables} if observables else {}),
        },
        "equations": {
            "model": [eq if ("=" in eq) else f"{eq} = 0" for eq in equations],
            **({
                "observables": {v: v for v in observables}
            } if observables else {}),
        },
        "calibration": {
            "parameters": param_values,
            "covariance": covariance,
        },
        # estimation is optional; leave empty
    }

    return model_yaml
