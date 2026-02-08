from __future__ import annotations

import re
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


_TIME_INDEX_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([+-]?\d+)\s*\)")


def _infer_max_lead_lag(equations: List[str], timed_names: set[str]) -> tuple[int, int]:
    max_lead = 1
    max_lag = 1
    for eq in equations:
        for name, idx_s in _TIME_INDEX_RE.findall(eq):
            if name not in timed_names:
                continue
            idx = int(idx_s)
            if idx > max_lead:
                max_lead = idx
            if idx < 0 and -idx > max_lag:
                max_lag = -idx
    return max_lead, max_lag


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

    timed_names = set(variables) | set(shocks)
    max_lead, max_lag = _infer_max_lead_lag(equations, timed_names)

    equations_list = [eq if ("=" in eq) else f"{eq} = 0" for eq in equations]

    # Match dsge's YAML schema conventions:
    # - omit `declarations.type` (defaults to dsge)
    # - allow `equations` to be either a bare list (model-only) or a dict with `model` + `observables`
    if observables:
        equations_block: Any = {
            "model": equations_list,
            "observables": {v: v for v in observables},
        }
    else:
        equations_block = equations_list

    model_yaml: Dict[str, Any] = {
        "declarations": {
            "name": name,
            "variables": variables,
            "parameters": parameters,
            "shocks": shocks,
            "max_lead": max_lead,
            "max_lag": max_lag,
            **({"observables": observables} if observables else {}),
        },
        "equations": equations_block,
        "calibration": {
            "parameters": param_values,
            "covariance": covariance,
        },
        # estimation is optional; leave empty
    }

    return model_yaml
