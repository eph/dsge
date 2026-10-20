"""Compile quadratic losses and squared one-sided penalties for policy solvers."""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import sympy


@dataclass(frozen=True)
class CompiledPolicyLoss:
    """Quadratic loss and inequalities, including independent auxiliary choices."""

    hessian: np.ndarray
    gradient: np.ndarray
    constraints: np.ndarray
    bounds: np.ndarray
    auxiliary_labels: tuple[str, ...]

    @property
    def auxiliary_count(self) -> int:
        return len(self.auxiliary_labels)


def compile_policy_loss(
    expression: sympy.Expr,
    variables: Sequence[sympy.Expr],
    evaluate: Callable,
) -> CompiledPolicyLoss:
    """Compile a convex quadratic plus nonnegative weighted squared hinges.

    A hinge must be ``Min(affine, 0)`` or ``Max(affine, 0)``. Each squared
    hinge introduces an independent auxiliary outcome and control. Minimizing
    its square subject to ``affine - auxiliary >= 0`` (Min) or the reversed
    inequality (Max) reproduces the original penalty exactly. These auxiliary
    controls have no effect on the physical model.
    """
    variables = list(variables)
    n = len(variables)
    expression = sympy.expand(expression)
    hinges = sorted(expression.atoms(sympy.Min, sympy.Max), key=str)
    penalties = []
    quadratic = expression
    for hinge in hinges:
        if len(hinge.args) != 2 or sympy.S.Zero not in hinge.args:
            raise ValueError("Squared min/max losses require one affine argument and zero.")
        affine = next(arg for arg in hinge.args if arg != 0)
        if affine.has(sympy.Min, sympy.Max, sympy.Piecewise):
            raise ValueError("Nested min/max losses are not supported; use a sum of squared min/max terms.")
        try:
            if sympy.Poly(affine, *variables, domain="EX").total_degree() > 1:
                raise ValueError("The argument of each squared min/max loss must be affine.")
        except sympy.PolynomialError as exc:
            raise ValueError("The argument of each squared min/max loss must be affine.") from exc
        coefficient = expression.coeff(hinge, 2)
        if (coefficient == 0 or coefficient.has(*variables)
                or coefficient.has(sympy.Min, sympy.Max, sympy.Piecewise)):
            raise ValueError(
                "Use a sum of nonnegative weighted squared min/max terms; "
                "products, unsquared terms, and variable-dependent weights are unsupported."
            )
        weight = float(np.asarray(evaluate(coefficient)))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Squared min/max loss weights must be finite and nonnegative.")
        quadratic -= coefficient * hinge**2
        if weight > 0:
            row = np.asarray(evaluate(sympy.Matrix([[affine.diff(v) for v in variables]])), float).reshape(-1)
            constant = float(np.asarray(evaluate(affine.subs({v: 0 for v in variables}))))
            penalties.append((hinge, weight, row, constant))
    quadratic = sympy.expand(quadratic)
    if quadratic.has(sympy.Min, sympy.Max, sympy.Piecewise):
        raise ValueError("Losses support only a quadratic plus nonnegative weighted squared min/max terms.")
    try:
        if sympy.Poly(quadratic, *variables, domain="EX").total_degree() > 2:
            raise ValueError("The remaining period loss must be quadratic (degree <= 2).")
    except sympy.PolynomialError as exc:
        raise ValueError("The remaining period loss must be a quadratic polynomial.") from exc
    hessian = np.asarray(evaluate(sympy.hessian(quadratic, variables)), float)
    gradient = np.asarray(evaluate(sympy.Matrix([quadratic.diff(v) for v in variables])
                                   .subs({v: 0 for v in variables})), float).reshape(-1)
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(gradient)):
        raise ValueError("The calibrated loss must be finite.")
    if np.linalg.eigvalsh(hessian).min(initial=0) < -1e-10 * max(1, np.linalg.norm(hessian)):
        raise ValueError("The quadratic part of the period loss must be convex.")
    count = len(penalties)
    W = np.zeros((n + count, n + count))
    W[:n, :n] = hessian
    g = np.r_[gradient, np.zeros(count)]
    C = np.zeros((count, n + count))
    bounds = np.zeros(count)
    labels = []
    for j, (hinge, weight, row, constant) in enumerate(penalties):
        sign = 1 if hinge.func == sympy.Min else -1
        W[n + j, n + j] = 2 * weight
        C[j, :n] = sign * row
        C[j, n + j] = -sign
        bounds[j] = -sign * constant
        labels.append(str(hinge))
    if not np.all(np.isfinite(C)) or not np.all(np.isfinite(bounds)):
        raise ValueError("Squared min/max loss arguments must have finite calibrated coefficients.")
    return CompiledPolicyLoss(W, g, C, bounds, tuple(labels))
