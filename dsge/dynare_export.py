from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import sympy
from lark import Lark, UnexpectedInput

from .codegen_safety import code_identifier, code_integer, single_line_text
from .safe_expression import parse_expression
from .symbols import Equation, Parameter, Shock, Variable


_FUNCTIONS = {
    f: name for name, f in {
        "exp": sympy.exp, "log": sympy.log, "sin": sympy.sin,
        "cos": sympy.cos, "tan": sympy.tan, "asin": sympy.asin,
        "acos": sympy.acos, "atan": sympy.atan, "sinh": sympy.sinh,
        "cosh": sympy.cosh, "tanh": sympy.tanh, "abs": sympy.Abs,
        "sign": sympy.sign, "erf": sympy.erf,
    }.items()
}

# This grammar describes only the source produced here. The permissive Dynare
# importer deliberately accepts/ignores other commands and is not a validator.
_GENERATED_PARSER = Lark(r"""
start: parameters? variables shocks_decl? model initval "steady" ";" shocks? simulation
parameters: "parameters" NAME+ ";" assignment*
variables: "var" NAME+ ";"
shocks_decl: "varexo" NAME+ ";"
model: "model" ";" equation+ "end" ";"
equation: expr "=" expr ";"
initval: "initval" ";" assignment+ "end" ";"
assignment: NAME "=" expr ";"
shocks: "shocks" ";" shock* "end" ";"
shock: "var" NAME ";" "stderr" expr ";"
     | "var" NAME "," NAME "=" expr ";"
simulation: "stoch_simul(order=" INT ("," flag)* ");"
flag: "pruning" | "irf=" INT | "periods=" INT
?expr: sum
?sum: product (("+"|"-") product)*
?product: power (("*"|"/") power)*
?power: atom ("^" atom)*
?atom: NUMBER | NAME | function | lag | "(" expr ")" | "-" atom
function: FUNCTION "(" expr ("," expr)* ")"
lag: NAME "(" SIGNED_INT ")"
FUNCTION.2: /(?:exp|log|sqrt|sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|abs|sign|erf)(?=\s*\()/
NAME: /[A-Za-z][A-Za-z0-9_]*/
NUMBER: /(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?/
%import common.INT
%import common.SIGNED_INT
%import common.WS
%ignore WS
%ignore /\/\/[^\r\n]*/
""", parser="lalr")


def validate_generated_dynare(text: str) -> None:
    """Reject any statement outside the generated model subset."""
    try:
        _GENERATED_PARSER.parse(text)
    except UnexpectedInput as exc:
        raise ValueError("Invalid generated Dynare source") from exc


@dataclass(frozen=True)
class DynareMod:
    mod_text: str
    variables: List[str]
    shocks: List[str]
    parameters: List[str]


def _expr_to_dynare(expr, context: Dict[str, object]) -> str:
    """Serialize a mathematical AST, never generic model-supplied source text."""
    if not isinstance(expr, sympy.Basic):
        expr = parse_expression(expr, context)

    def emit(node):
        if isinstance(node, sympy.Symbol):
            name = code_identifier(node.name, language="Dynare")
            if name not in context or not isinstance(context[name], sympy.Symbol):
                raise ValueError(f"Undeclared Dynare symbol: {name!r}")
            if isinstance(node, (Variable, Shock)):
                date = code_integer(node.date, field="Dynare symbol date")
                if getattr(node, "exp_date", 0) != 0:
                    raise ValueError("Dynare export does not support dated expectations")
                return name if date == 0 else f"{name}({date})"
            return name
        if isinstance(node, sympy.Integer):
            return str(int(node))
        if isinstance(node, sympy.Rational):
            return f"({int(node.p)}/{int(node.q)})"
        if isinstance(node, sympy.Float) and node.is_finite:
            return str(node)
        if node in (sympy.pi, sympy.E):
            return str(node.evalf(17))
        if isinstance(node, (sympy.Add, sympy.Mul)):
            op = " + " if isinstance(node, sympy.Add) else " * "
            return "(" + op.join(emit(arg) for arg in node.args) + ")"
        if isinstance(node, sympy.Pow):
            return f"({emit(node.base)}^{emit(node.exp)})"
        if node.func in _FUNCTIONS:
            return _FUNCTIONS[node.func] + "(" + ", ".join(emit(a) for a in node.args) + ")"
        raise ValueError(f"Unsupported Dynare expression node: {type(node).__name__}")

    return emit(expr)


def _equation_to_dynare(eq: Equation, context: Dict[str, object]) -> str:
    return f"{_expr_to_dynare(eq.lhs, context)} = {_expr_to_dynare(eq.rhs, context)};"


def _cov_to_dynare_shocks_block(
    shock_names: Sequence[str],
    covariance: Dict[str, object],
    context: Dict[str, object],
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
        if not isinstance(k, str):
            raise ValueError("Dynare covariance keys must be shock names")
        names = [code_identifier(x.strip(), language="Dynare") for x in k.split(",")]
        if len(names) not in (1, 2) or any(name not in shock_names for name in names):
            raise ValueError(f"Invalid Dynare covariance key: {k!r}")
        v_str = _expr_to_dynare(v, context)
        if len(names) == 2:
            s1, s2 = names
            off.append((s1, s2, v_str))
        else:
            diag[names[0]] = v_str

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

    var_names = [code_identifier(v.name, language="Dynare") for v in variables]
    shock_names = [code_identifier(s.name, language="Dynare") for s in shocks]
    param_names = [code_identifier(p.name, language="Dynare") for p in parameters]
    names = var_names + shock_names + param_names
    if len(set(names)) != len(names):
        raise ValueError("Dynare declarations must have distinct names")
    context = {name: function for function, name in _FUNCTIONS.items()}
    context.update({"sqrt": sympy.sqrt, "Abs": sympy.Abs})
    context.update({s.name: s for s in variables + shocks + parameters})
    parameter_context = {k: v for k, v in context.items() if k not in var_names + shock_names}
    order = code_integer(order, field="Dynare order")
    periods = code_integer(periods, field="Dynare periods")
    if irf is not None:
        irf = code_integer(irf, field="Dynare irf")
    if order < 1 or periods < 0 or (irf is not None and irf < 0):
        raise ValueError("Dynare order must be positive and periods/irf nonnegative")

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
    name = single_line_text(model.get("name", "model"), field="Model name")
    lines.append(f"// Auto-generated from dsge YAML model: {name}")
    lines.append("")
    if param_names:
        lines.append("parameters " + " ".join(param_names) + ";")
        for p in param_names:
            if str(p) in cal_params:
                lines.append(f"{p} = {_expr_to_dynare(cal_params[p], parameter_context)};")
            elif p in {s.name for s in model["auxiliary_parameters"]}:
                value = next(v for s, v in model["auxiliary_parameters"].items() if s.name == p)
                lines.append(f"{p} = {_expr_to_dynare(value, parameter_context)};")
        lines.append("")

    lines.append("var " + " ".join(var_names) + ";")
    if shock_names:
        lines.append("varexo " + " ".join(shock_names) + ";")
    lines.append("")

    lines.append("model;")
    for eq in eqs:
        lines.append("  " + _equation_to_dynare(eq, context))
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
        lines.append(_cov_to_dynare_shocks_block(shock_names, cov_dict, parameter_context))
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

    mod_text = "\n".join(lines)
    validate_generated_dynare(mod_text)
    return DynareMod(
        mod_text=mod_text,
        variables=var_names,
        shocks=shock_names,
        parameters=param_names,
    )
