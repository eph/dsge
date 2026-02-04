from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import sympy

from .symbols import Equation, Parameter, Shock, Variable


@dataclass(frozen=True)
class DynareMod:
    mod_text: str
    variables: List[str]
    shocks: List[str]
    parameters: List[str]


def _expr_to_dynare(expr: sympy.Expr) -> str:
    # Use sympy's printer (Variable.__str__ gives `x` or `x(-1)` in non-greek mode),
    # but convert Python power `**` back to Dynare `^`.
    s = sympy.sstr(expr)
    return s.replace("**", "^")


def _equation_to_dynare(eq: Equation) -> str:
    return f"{_expr_to_dynare(eq.lhs)} = {_expr_to_dynare(eq.rhs)};"


def _cov_to_dynare_shocks_block(
    shock_names: Sequence[str],
    covariance: Dict[str, object],
) -> str:
    """
    Convert YAML-style covariance dict into a Dynare shocks block.

    Supports:
    - diagonal entries: {"e": 0.01} or {"e": "(0.1)**2"}
    - off-diagonals keyed as "e,u": value expression (treated as covariance, not correlation)
    """
    diag: Dict[str, str] = {}
    off: List[Tuple[str, str, str]] = []
    for k, v in covariance.items():
        v_str = str(v).replace("**", "^")
        if "," in k:
            s1, s2 = [x.strip() for x in k.split(",", 1)]
            off.append((s1, s2, v_str))
        else:
            diag[k.strip()] = v_str

    lines: List[str] = ["shocks;"]
    for s in shock_names:
        if s in diag:
            lines.append(f"  var {s}; stderr sqrt({diag[s]});")
    for s1, s2, cov_expr in off:
        if s1 in shock_names and s2 in shock_names:
            lines.append(f"  var {s1}, {s2} = {cov_expr};")
    lines.append("end;")
    return "\n".join(lines)


def _covariance_to_dict(covariance, shock_names: Sequence[str]) -> Dict[str, object]:
    if isinstance(covariance, dict):
        return covariance
    if isinstance(covariance, (sympy.MatrixBase,)):
        cov_dict: Dict[str, object] = {}
        for i, s1 in enumerate(shock_names):
            cov_dict[s1] = covariance[i, i]
            for j, s2 in enumerate(shock_names):
                if j <= i:
                    continue
                v = covariance[i, j]
                if v != 0:
                    cov_dict[f"{s1},{s2}"] = v
        return cov_dict
    raise TypeError(f"Unsupported covariance type: {type(covariance)}")


def to_dynare_mod(
    model,
    *,
    order: int = 2,
    pruning: bool = True,
    irf: int | None = 0,
    periods: int = 0,
    drop_re_errors: bool = True,
) -> DynareMod:
    """
    Export a DSGE model to a Dynare .mod text (best-effort).

    Notes
    -----
    - Intended for LRE `DSGE` models expressed in deviations with steady state at 0.
    - If `drop_re_errors=True`, the internal `eta_*` rational expectations shocks
      introduced by the Python LRE plumbing are omitted from the exported shocks list.
    """
    eqs: Sequence[Equation] = list(model["perturb_eq"])
    variables: List[Variable] = list(model["var_ordering"])
    parameters: List[Parameter] = list(model["parameters"]) + list(model["auxiliary_parameters"].keys())
    shocks: List[Shock] = list(model["shk_ordering"])

    if drop_re_errors and "re_errors" in model:
        re_error_names = {s.name for s in model["re_errors"]}
        shocks = [s for s in shocks if s.name not in re_error_names]

    var_names = [v.name for v in variables]
    shock_names = [s.name for s in shocks]
    param_names = [p.name for p in parameters]

    calibration = model.get("calibration", {})
    if isinstance(calibration, dict) and "parameters" in calibration and isinstance(calibration["parameters"], dict):
        cal_params: Dict[str, object] = calibration["parameters"]
    elif isinstance(calibration, dict):
        # DSGE models store calibration as a flat {param_name: value} dict.
        cal_params = calibration
    else:
        cal_params = {}
    cov = model.get("covariance", {})

    lines: List[str] = []
    lines.append(f"// Auto-generated from dsge YAML model: {model.get('name', 'model')}")
    lines.append("")
    if param_names:
        lines.append("parameters " + " ".join(param_names) + ";")
        for p in param_names:
            if str(p) in cal_params:
                lines.append(f"{p} = {str(cal_params[str(p)]).replace('**', '^')};")
        lines.append("")

    lines.append("var " + " ".join(var_names) + ";")
    if shock_names:
        lines.append("varexo " + " ".join(shock_names) + ";")
    lines.append("")

    lines.append("model;")
    for eq in eqs:
        lines.append("  " + _equation_to_dynare(eq))
    lines.append("end;")
    lines.append("")

    # Default steady state and init values: zero (deviation form).
    lines.append("initval;")
    for v in var_names:
        lines.append(f"  {v} = 0;")
    lines.append("end;")
    lines.append("")
    lines.append("steady;")
    lines.append("")

    if shock_names:
        cov_dict = _covariance_to_dict(cov, shock_names)
        lines.append(_cov_to_dynare_shocks_block(shock_names, cov_dict))
        lines.append("")

    stoch_flags = []
    if pruning and order >= 2:
        stoch_flags.append("pruning")
    if irf is not None:
        stoch_flags.append(f"irf={int(irf)}")
    if periods and periods > 0:
        stoch_flags.append(f"periods={int(periods)}")
    flags = (", " + ", ".join(stoch_flags)) if stoch_flags else ""
    lines.append(f"stoch_simul(order={int(order)}{flags});")
    lines.append("")

    return DynareMod(
        mod_text="\n".join(lines),
        variables=var_names,
        shocks=shock_names,
        parameters=param_names,
    )
