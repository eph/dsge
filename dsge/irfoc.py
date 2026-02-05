from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import sympy

from .logging_config import get_logger
from .symbols import Equation, Variable

logger = get_logger("dsge.irfoc")


@dataclass(frozen=True)
class IRFOCResult:
    """
    Result container for IRFOC simulations.
    """

    simulation: pd.DataFrame
    shocks: pd.DataFrame
    residuals: pd.DataFrame


def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _split_equation(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise ValueError("Policy rule must contain '=' (e.g. 'i = 1.5*pi').")
    lhs, rhs = s.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if not lhs or not rhs:
        raise ValueError(f"Invalid policy rule equation: {s!r}.")
    return lhs, rhs


def _sympify_with_context(expr: str, context: dict[str, object]) -> sympy.Expr:
    # Use SymPy's parser with a controlled locals dict (no eval of Python names).
    return sympy.sympify(expr, locals=context)


def parse_policy_rules(
    rules: str | Sequence[str],
    *,
    variables: Sequence[sympy.Expr],
    variable_context: dict[str, object],
) -> list[Equation]:
    """
    Parse one or more policy-rule equations into `Equation` objects.

    Parameters
    ----------
    rules
        Either a single string `'lhs = rhs'` or a list of such strings.
    variables
        The variables that rules are allowed to reference (used later for affine checks).
    variable_context
        Locals dict passed to `sympy.sympify` (should map variable names to dsge `Variable` objects).
    """
    rule_list = _as_list(rules)
    if not rule_list:
        raise ValueError("At least one policy rule equation is required.")

    context = dict(variable_context)
    context.update({"max": sympy.Max, "min": sympy.Min})

    eqs: list[Equation] = []
    for r in rule_list:
        lhs_s, rhs_s = _split_equation(str(r))
        lhs = _sympify_with_context(lhs_s, context)
        rhs = _sympify_with_context(rhs_s, context)
        eqs.append(Equation(lhs, rhs))

    # Quick sanity: disallow piecewise max/min rules for now.
    for eq in eqs:
        if eq.set_eq_zero.has(sympy.Max, sympy.Min, sympy.Piecewise):
            raise NotImplementedError(
                "IRFOC currently supports only affine (linear/constant) policy rules. "
                "Rules with max/min/Piecewise are not implemented yet."
            )

    # Affine check relative to the allowed variables.
    for eq in eqs:
        try:
            poly = sympy.Poly(eq.set_eq_zero, *variables, domain="EX")
            if poly.total_degree() > 1:
                raise ValueError(f"Policy rule must be affine in model variables, got: {str(eq)}")
        except sympy.PolynomialError:
            raise ValueError(f"Policy rule must be affine in model variables, got: {str(eq)}") from None

    return eqs


class IRFOC:
    """
    IRF-based optimal-control / counterfactual simulator (Fabian-James style).

    This class takes a baseline path (typically an IRF or any deterministic simulation)
    and computes the sequence of an "instrument shock" needed to enforce an affine
    policy rule period-by-period.

    The core linear algebra solves:

        Ω (y_baseline + M u) + 1⊗b = 0

    where:
    - y_baseline is the baseline path (stacked over time),
    - u is the instrument-shock path (stacked over time),
    - M maps instrument shocks into the endogenous paths (using anticipated IRFs),
    - Ω applies the per-period policy-rule restrictions,
    - b is the per-period constant term from the affine rule.
    """

    def __init__(
        self,
        model,
        baseline: pd.DataFrame,
        instrument_shocks: str | Sequence[str],
        *,
        p0: Sequence[float] | None = None,
        compiled_model=None,
    ):
        if not isinstance(baseline, pd.DataFrame):
            raise TypeError("baseline must be a pandas.DataFrame")
        if baseline.empty:
            raise ValueError("baseline must be non-empty")

        self.model = model
        self.baseline = baseline.copy()

        self.T, self.n = self.baseline.shape
        self.columns = list(self.baseline.columns)

        self.instrument_shocks = [str(s) for s in _as_list(instrument_shocks)]
        if not self.instrument_shocks:
            raise ValueError("At least one instrument shock must be provided.")

        if p0 is None:
            try:
                p0 = model.p0()
            except Exception:
                p0 = None
        self.p0 = p0

        self.compiled_model = compiled_model if compiled_model is not None else model.compile_model()

        # Context for parsing column names and rules.
        model_vars = list(self.model["var_ordering"])
        context = {v.name: v for v in model_vars}
        # Also allow any auxiliary variables to be referenced by plain name if present in the model.
        for v in model_vars:
            if isinstance(v, Variable):
                context.setdefault(str(v), v)

        self._variable_context = context
        self._baseline_variables = [sympy.sympify(str(c), locals=context) for c in self.columns]

        self.M = self._build_M()

    @property
    def baseline_variables(self) -> list[sympy.Expr]:
        return list(self._baseline_variables)

    def _build_M(self) -> np.ndarray:
        if self.p0 is None:
            raise ValueError("p0 must be provided (or model must support model.p0()).")

        blocks = []
        for shock in self.instrument_shocks:
            irf0 = self.compiled_model.impulse_response(self.p0, h=self.T - 1)[shock].loc[:, self.columns]
            cols = [irf0.values.reshape(-1)]
            for t in range(1, self.T):
                irft = self.compiled_model.anticipated_impulse_response(
                    self.p0, anticipated_h=t, h=self.T - 1, use_cache=True
                )[shock].loc[:, self.columns]
                cols.append(irft.values.reshape(-1))
            blocks.append(np.column_stack(cols))

        return np.column_stack(blocks) if len(blocks) > 1 else blocks[0]

    def _affine_A_b(self, rules: str | Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[Equation]]:
        eqs = parse_policy_rules(
            rules,
            variables=self._baseline_variables,
            variable_context=self._variable_context,
        )

        ninstr = len(eqs)
        nv = len(self._baseline_variables)
        A = sympy.Matrix(ninstr, nv, lambda i, j: eqs[i].set_eq_zero.diff(self._baseline_variables[j]))
        b = sympy.Matrix(ninstr, 1, lambda i, _: eqs[i].set_eq_zero.subs({v: 0 for v in self._baseline_variables}))

        return np.asarray(A, dtype=float), np.asarray(b, dtype=float).reshape(-1), eqs

    def simulate(
        self,
        rules: str | Sequence[str],
        *,
        rcond: float | None = None,
        return_details: bool = False,
    ) -> pd.DataFrame | IRFOCResult:
        """
        Simulate an alternative path consistent with an affine policy rule.

        Parameters
        ----------
        rules
            `'lhs = rhs'` string or a list of rule strings.
        rcond
            Passed to `np.linalg.lstsq` if the linear system is singular.
        return_details
            If True, return an `IRFOCResult` including the implied instrument shocks and residuals.
        """
        A, b, eqs = self._affine_A_b(rules)

        if A.shape[0] * self.T != self.M.shape[1]:
            raise ValueError(
                "System is not square: (#rules * T) must equal (#instrument_shocks * T). "
                f"Got rules={A.shape[0]}, T={self.T}, instrument_shocks={len(self.instrument_shocks)}."
            )

        omega_y = np.kron(np.eye(self.T), A)
        omega_M = omega_y @ self.M

        y0 = self.baseline.values.reshape(-1)
        rhs = -(omega_y @ y0 + np.kron(np.ones(self.T), b))

        try:
            u = np.linalg.solve(omega_M, rhs)
        except np.linalg.LinAlgError:
            u, *_ = np.linalg.lstsq(omega_M, rhs, rcond=rcond)

        y_alt = y0 + self.M @ u
        sim = pd.DataFrame(y_alt.reshape(self.T, self.n), columns=self.columns, index=self.baseline.index)

        if not return_details:
            return sim

        shocks = self._u_to_df(u)
        resid = self._residuals(sim, A, b, eqs)
        return IRFOCResult(simulation=sim, shocks=shocks, residuals=resid)

    def _u_to_df(self, u: np.ndarray) -> pd.DataFrame:
        u = np.asarray(u, dtype=float).reshape(-1)
        k = len(self.instrument_shocks)
        if k == 1:
            arr = u.reshape(self.T, 1)
            cols = [self.instrument_shocks[0]]
        else:
            arr = u.reshape(self.T, k, order="F")
            cols = list(self.instrument_shocks)
        return pd.DataFrame(arr, columns=cols, index=self.baseline.index)

    def _residuals(self, sim: pd.DataFrame, A: np.ndarray, b: np.ndarray, eqs: Iterable[Equation]) -> pd.DataFrame:
        y = sim.values
        res = y @ A.T + b[None, :]
        names = [str(eq.set_eq_zero) for eq in eqs]
        return pd.DataFrame(res, columns=names, index=sim.index)

    # Back-compat with the name used in notes.
    def sim_simple_policy_rule(self, rule: str | Sequence[str]) -> pd.DataFrame:
        return self.simulate(rule, return_details=False)


# Alias to match the class name in the user's notes.
IRFBasedCounterfactual = IRFOC

