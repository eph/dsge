from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import sympy
from scipy.optimize import Bounds, LinearConstraint, milp

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
    allow_piecewise: bool = False,
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

    if not allow_piecewise:
        # Quick sanity: disallow piecewise max/min rules.
        for eq in eqs:
            if eq.set_eq_zero.has(sympy.Max, sympy.Min, sympy.Piecewise):
                raise NotImplementedError(
                    "IRFOC currently supports only affine (linear/constant) policy rules. "
                    "Rules with max/min/Piecewise are available via `IRFOC.simulate_piecewise()`."
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
            allow_piecewise=False,
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
        # Dispatch to MILP backend if rules contain max/min/Piecewise.
        try:
            eqs_pw = parse_policy_rules(
                rules,
                variables=self._baseline_variables,
                variable_context=self._variable_context,
                allow_piecewise=True,
            )
        except Exception:
            eqs_pw = None

        if eqs_pw is not None and any(eq.set_eq_zero.has(sympy.Max, sympy.Min, sympy.Piecewise) for eq in eqs_pw):
            return self.simulate_piecewise(rules, return_details=return_details)

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

    class _LinExpr:
        def __init__(self, a_u: np.ndarray, a_w: dict[int, float] | None = None, c: float = 0.0):
            self.a_u = np.asarray(a_u, dtype=float).reshape(-1)
            self.a_w = {} if a_w is None else dict(a_w)
            self.c = float(c)

        def add(self, other: "IRFOC._LinExpr") -> "IRFOC._LinExpr":
            a_u = self.a_u + other.a_u
            a_w = dict(self.a_w)
            for k, v in other.a_w.items():
                a_w[k] = a_w.get(k, 0.0) + float(v)
            return IRFOC._LinExpr(a_u, a_w, self.c + other.c)

        def scale(self, s: float) -> "IRFOC._LinExpr":
            s = float(s)
            a_u = s * self.a_u
            a_w = {k: s * v for k, v in self.a_w.items()}
            return IRFOC._LinExpr(a_u, a_w, s * self.c)

        def to_continuous_vector(self, nu: int, nw: int) -> np.ndarray:
            out = np.zeros(nu + nw, dtype=float)
            out[:nu] = self.a_u
            for k, v in self.a_w.items():
                out[nu + k] += float(v)
            return out

        def bounds(self, u_lb: np.ndarray, u_ub: np.ndarray, w_bounds: list[tuple[float, float]]) -> tuple[float, float]:
            lo = self.c + float(np.sum(np.where(self.a_u >= 0, self.a_u * u_lb, self.a_u * u_ub)))
            hi = self.c + float(np.sum(np.where(self.a_u >= 0, self.a_u * u_ub, self.a_u * u_lb)))
            for k, v in self.a_w.items():
                wl, wu = w_bounds[k]
                if v >= 0:
                    lo += v * wl
                    hi += v * wu
                else:
                    lo += v * wu
                    hi += v * wl
            return float(lo), float(hi)

    def simulate_piecewise(
        self,
        rules: str | Sequence[str],
        *,
        u_bounds: tuple[float, float] | dict[str, tuple[float, float]] = (-10.0, 10.0),
        objective: str = "min_l1_shocks",
        return_details: bool = False,
    ) -> pd.DataFrame | IRFOCResult:
        """
        Enforce piecewise-affine rules via MILP.

        Currently supports nested `max()`/`min()` over affine expressions of baseline variables.
        """
        eqs = parse_policy_rules(
            rules,
            variables=self._baseline_variables,
            variable_context=self._variable_context,
            allow_piecewise=True,
        )
        if len(eqs) != len(self.instrument_shocks):
            raise ValueError(
                "Piecewise solve requires #rules == #instrument_shocks. "
                f"Got rules={len(eqs)}, instrument_shocks={len(self.instrument_shocks)}."
            )

        nu = int(self.M.shape[1])
        u_lb, u_ub = self._expand_u_bounds(u_bounds)

        use_l1 = str(objective).lower() in {"min_l1_shocks", "min_l1", "l1"}
        if str(objective).lower() not in {"feasible", "min_l1_shocks", "min_l1", "l1"}:
            raise ValueError("objective must be 'feasible' or 'min_l1_shocks'.")

        # Max-node records: (w_idx, left_expr, right_expr)
        max_nodes: list[tuple[int, IRFOC._LinExpr, IRFOC._LinExpr]] = []

        def baseline_var_expr(var: sympy.Expr, t: int) -> "IRFOC._LinExpr":
            j = self._baseline_variables.index(var)
            a_u = np.asarray(self.M[t * self.n + j, :], dtype=float)
            c = float(self.baseline.iloc[t, j])
            return IRFOC._LinExpr(a_u=a_u, c=c)

        def const_expr(x: float) -> "IRFOC._LinExpr":
            return IRFOC._LinExpr(a_u=np.zeros(nu, dtype=float), c=float(x))

        def compile_lin(expr: sympy.Expr, t: int) -> "IRFOC._LinExpr":
            if expr.is_number:
                return const_expr(float(expr))
            if expr in self._baseline_variables:
                return baseline_var_expr(expr, t)

            if expr.func is sympy.Add:
                out = const_expr(0.0)
                for arg in expr.args:
                    out = out.add(compile_lin(arg, t))
                return out

            if expr.func is sympy.Mul:
                scalars = []
                non_scalars = []
                for arg in expr.args:
                    if arg.is_number:
                        scalars.append(float(arg))
                    else:
                        non_scalars.append(arg)
                scale = float(np.prod(scalars)) if scalars else 1.0
                if len(non_scalars) == 0:
                    return const_expr(scale)
                if len(non_scalars) != 1:
                    raise ValueError("Only scalar * affine is supported in piecewise rules.")
                return compile_lin(non_scalars[0], t).scale(scale)

            if expr.func is sympy.Pow:
                base, power = expr.args
                if power == 1:
                    return compile_lin(base, t)
                raise ValueError("Piecewise rules support only affine expressions (no powers).")

            if expr.func in (sympy.Max, sympy.Min):
                args = list(expr.args)
                if len(args) < 2:
                    raise ValueError("max/min requires at least two arguments.")
                if len(args) > 2:
                    cur = sympy.Max(args[0], args[1]) if expr.func is sympy.Max else sympy.Min(args[0], args[1])
                    for nxt in args[2:]:
                        cur = sympy.Max(cur, nxt) if expr.func is sympy.Max else sympy.Min(cur, nxt)
                    return compile_lin(cur, t)

                if expr.func is sympy.Min:
                    return compile_lin(-sympy.Max(-args[0], -args[1]), t)

                left = compile_lin(args[0], t)
                right = compile_lin(args[1], t)
                w_idx = len(max_nodes)
                max_nodes.append((w_idx, left, right))
                return IRFOC._LinExpr(a_u=np.zeros(nu, dtype=float), a_w={w_idx: 1.0}, c=0.0)

            raise ValueError(f"Unsupported function in piecewise rule: {expr.func}")

        # Build constraints by time and rule.
        eq_forms: list[IRFOC._LinExpr] = []
        for t in range(self.T):
            for eq in eqs:
                lhs = compile_lin(eq.lhs, t)
                rhs = compile_lin(eq.rhs, t)
                eq_forms.append(lhs.add(rhs.scale(-1.0)))

        nw = len(max_nodes)
        nb = nw
        ns = nu if use_l1 else 0
        offset_u = 0
        offset_s = nu
        offset_w = nu + ns
        offset_z = nu + ns + nw
        nvar = nu + ns + nw + nb

        w_bounds: list[tuple[float, float]] = []
        # Compute bounds for each w node in creation order (postorder by construction).
        for w_idx, left, right in max_nodes:
            aL, aU = left.bounds(u_lb, u_ub, w_bounds)
            bL, bU = right.bounds(u_lb, u_ub, w_bounds)
            w_bounds.append((max(aL, bL), max(aU, bU)))

        constraints: list[LinearConstraint] = []

        # Equality constraints for the rule(s): form == 0.
        for form in eq_forms:
            A = np.zeros((1, nvar), dtype=float)
            cont = form.to_continuous_vector(nu, nw)
            A[0, offset_u : offset_u + nu] = cont[:nu]
            if nw:
                A[0, offset_w : offset_w + nw] = cont[nu:]
            constraints.append(LinearConstraint(A, lb=[-form.c], ub=[-form.c]))

        # L1 objective constraints: s >= u and s >= -u.
        if use_l1:
            A1 = np.zeros((nu, nvar), dtype=float)
            A1[:, offset_s : offset_s + nu] = np.eye(nu)
            A1[:, offset_u : offset_u + nu] = -np.eye(nu)
            constraints.append(LinearConstraint(A1, lb=np.zeros(nu), ub=np.inf * np.ones(nu)))

            A2 = np.zeros((nu, nvar), dtype=float)
            A2[:, offset_s : offset_s + nu] = np.eye(nu)
            A2[:, offset_u : offset_u + nu] = np.eye(nu)
            constraints.append(LinearConstraint(A2, lb=np.zeros(nu), ub=np.inf * np.ones(nu)))

        # Max node constraints (Big-M).
        for w_idx, left, right in max_nodes:
            z_idx = w_idx

            # w - left >= 0, w - right >= 0
            for operand in (left, right):
                A = np.zeros((1, nvar), dtype=float)
                cont = operand.to_continuous_vector(nu, nw)
                A[0, offset_w + w_idx] = 1.0
                A[0, offset_u : offset_u + nu] = -cont[:nu]
                if nw:
                    A[0, offset_w : offset_w + nw] += -cont[nu:]
                constraints.append(LinearConstraint(A, lb=[operand.c], ub=[np.inf]))

            aL, aU = left.bounds(u_lb, u_ub, w_bounds)
            bL, bU = right.bounds(u_lb, u_ub, w_bounds)
            M1 = max(float(bU - aL), 0.0)
            M2 = max(float(aU - bL), 0.0)

            # w - left <= M1*(1-z)  -> w - left + M1*z <= left.c + M1
            A = np.zeros((1, nvar), dtype=float)
            cont = left.to_continuous_vector(nu, nw)
            A[0, offset_w + w_idx] = 1.0
            A[0, offset_u : offset_u + nu] = -cont[:nu]
            if nw:
                A[0, offset_w : offset_w + nw] += -cont[nu:]
            A[0, offset_z + z_idx] = M1
            constraints.append(LinearConstraint(A, lb=[-np.inf], ub=[left.c + M1]))

            # w - right <= M2*z  -> w - right - M2*z <= right.c
            A = np.zeros((1, nvar), dtype=float)
            cont = right.to_continuous_vector(nu, nw)
            A[0, offset_w + w_idx] = 1.0
            A[0, offset_u : offset_u + nu] = -cont[:nu]
            if nw:
                A[0, offset_w : offset_w + nw] += -cont[nu:]
            A[0, offset_z + z_idx] = -M2
            constraints.append(LinearConstraint(A, lb=[-np.inf], ub=[right.c]))

        c = np.zeros(nvar, dtype=float)
        if use_l1:
            c[offset_s : offset_s + nu] = 1.0

        integrality = np.zeros(nvar, dtype=int)
        if nb:
            integrality[offset_z : offset_z + nb] = 1

        lb = np.full(nvar, -np.inf, dtype=float)
        ub = np.full(nvar, np.inf, dtype=float)
        lb[offset_u : offset_u + nu] = u_lb
        ub[offset_u : offset_u + nu] = u_ub
        if use_l1:
            lb[offset_s : offset_s + nu] = 0.0
        if nw:
            for i, (wl, wu) in enumerate(w_bounds):
                lb[offset_w + i] = wl
                ub[offset_w + i] = wu
        if nb:
            lb[offset_z : offset_z + nb] = 0.0
            ub[offset_z : offset_z + nb] = 1.0

        res = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=constraints,
            options={"disp": False},
        )
        if res.x is None or res.status != 0:
            raise RuntimeError(f"MILP failed: status={res.status}, message={getattr(res, 'message', None)}")

        u = np.asarray(res.x[offset_u : offset_u + nu], dtype=float)
        y0 = self.baseline.values.reshape(-1)
        y_alt = y0 + self.M @ u
        sim = pd.DataFrame(y_alt.reshape(self.T, self.n), columns=self.columns, index=self.baseline.index)

        if not return_details:
            return sim

        shocks = self._u_to_df(u)
        resid = pd.DataFrame(index=sim.index)
        return IRFOCResult(simulation=sim, shocks=shocks, residuals=resid)

    def _expand_u_bounds(
        self, u_bounds: tuple[float, float] | dict[str, tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        nu = int(self.M.shape[1])
        if isinstance(u_bounds, dict):
            out_l: list[float] = []
            out_u: list[float] = []
            bounds_by_shock = {str(k): v for k, v in u_bounds.items()}
            for shock in self.instrument_shocks:
                if shock not in bounds_by_shock:
                    raise ValueError(f"Missing bounds for instrument shock {shock!r}.")
                lo, hi = bounds_by_shock[shock]
                out_l.extend([float(lo)] * self.T)
                out_u.extend([float(hi)] * self.T)
            return np.asarray(out_l, dtype=float), np.asarray(out_u, dtype=float)

        lo, hi = u_bounds
        return float(lo) * np.ones(nu), float(hi) * np.ones(nu)


# Alias to match the class name in the user's notes.
IRFBasedCounterfactual = IRFOC
