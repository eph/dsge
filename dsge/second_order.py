from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sympy
from scipy.linalg import ordqz

from .logging_config import get_logger
from .symbols import Equation, Shock, Variable

logger = get_logger("dsge.second_order")


@dataclass(frozen=True)
class SecondOrderSolution:
    state_names: List[str]
    control_names: List[str]
    shock_names: List[str]

    hx: np.ndarray  # (nx, nx)
    gx: np.ndarray  # (ny, nx)
    hu: np.ndarray  # (nx, nu)
    gu: np.ndarray  # (ny, nu)

    hxx: np.ndarray  # (nx, nx, nx)
    gxx: np.ndarray  # (ny, nx, nx)
    hxu: np.ndarray  # (nx, nx, nu)
    gxu: np.ndarray  # (ny, nx, nu)
    huu: np.ndarray  # (nx, nu, nu)
    guu: np.ndarray  # (ny, nu, nu)

    hss: np.ndarray  # (nx,)
    gss: np.ndarray  # (ny,)

    shock_cov: np.ndarray  # (nu, nu)

    def as_dynare_like(self) -> Dict[str, object]:
        """
        Export decision-rule objects in a Dynare-like layout.

        Notes
        -----
        - Shapes follow Dynare conventions: `ghxx` is `(nendo, nstate^2)` using
          column-major vec over `(state, state)`, and similarly for `ghxu`, `ghuu`.
        - `ghs2` is `(nendo, 1)` and matches Dynare's `oo_.dr.ghs2` convention.
        - `state_names` are internal; for direct Dynare comparison you may need to map
          them to the corresponding lagged endogenous variables in Dynare.
        """
        ny, nx = self.gx.shape
        nu = self.gu.shape[1]

        ghxx = self.gxx.reshape(ny, nx * nx, order="F")
        # Dynare flattens `ghxu`/`ghuu` in a state-first / shock-first manner, which matches
        # NumPy's row-major ("C") order for the trailing dimensions.
        ghxu = self.gxu.reshape(ny, nx * nu, order="C")
        ghuu = self.guu.reshape(ny, nu * nu, order="C")
        ghs2 = self.gss.reshape(ny, 1)

        return {
            "endo_names": list(self.control_names),
            "state_names": list(self.state_names),
            "exo_names": list(self.shock_names),
            "ghx": np.asarray(self.gx, dtype=float),
            "ghu": np.asarray(self.gu, dtype=float),
            "ghxx": np.asarray(ghxx, dtype=float),
            "ghxu": np.asarray(ghxu, dtype=float),
            "ghuu": np.asarray(ghuu, dtype=float),
            "ghs2": np.asarray(ghs2, dtype=float),
        }


def _make_unique_name(existing: Iterable[str], base: str) -> str:
    existing_set = set(existing)
    if base not in existing_set:
        return base
    i = 2
    while f"{base}_{i}" in existing_set:
        i += 1
    return f"{base}_{i}"


def _numeric_context_for_model(model) -> Dict[str, object]:
    context_f: Dict[str, object] = {
        "exp": np.exp,
        "log": np.log,
    }

    # Keep parity with DSGE.python_sims_matrices external function support.
    try:
        external = model["__data__"]["declarations"].get("external")
    except Exception:
        external = None

    if external is not None:
        from importlib.machinery import SourceFileLoader
        import importlib.util

        f = external["file"]
        spec = importlib.util.spec_from_loader("external", SourceFileLoader("external", f))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for n in external["names"]:
            context_f[n] = getattr(module, n)

    # Optional convenience: normcdf if it appears in a user model.
    try:
        from scipy.stats import norm

        context_f.setdefault("normcdf", norm.cdf)
    except Exception:
        pass

    return context_f


@dataclass
class _SecondOrderCompiled:
    x0: List[Variable]
    y0: List[Variable]
    shocks: List[Shock]
    nx: int
    ny: int
    nu: int
    n: int
    tensor_entries: int
    v: List[object]
    eval_F1: Callable[[Sequence[float]], np.ndarray]
    eval_F0: Callable[[Sequence[float]], np.ndarray]
    eval_Fu: Callable[[Sequence[float]], np.ndarray]
    eval_Hi: List[Callable[[Sequence[float]], np.ndarray]]
    eval_QQ: Callable[[Sequence[float]], np.ndarray]


def _compile_second_order(model) -> _SecondOrderCompiled:
    """
    Compile symbolic derivatives needed for second-order perturbation once per model.

    The compiled object is parameterized only by the model's parameter vector `p`.
    """
    context_f = _numeric_context_for_model(model)

    base_vars = list(model["var_ordering"])
    shocks = list(model["shk_ordering"])
    equations = list(model["perturb_eq"])

    var_ordering, eqs, state_vars = _eliminate_lagged_endogenous(base_vars, equations)

    x0 = list(state_vars)
    y0 = [v for v in var_ordering if v not in state_vars]

    nx = len(x0)
    ny = len(y0)
    nu = len(shocks)
    n = nx + ny

    if n == 0:
        raise ValueError("Model has no endogenous variables.")
    if len(eqs) != n:
        raise ValueError(f"Expected {n} equations after lag elimination, got {len(eqs)}.")
    if nx == 0:
        raise NotImplementedError(
            "Second-order solver currently requires at least one lagged endogenous (a non-empty state)."
        )

    xp = [v(1) for v in x0]
    yp = [v(1) for v in y0]

    # Steady-state point: all endogenous and shocks at 0.
    subs_ss: Dict[object, object] = {}
    for v in var_ordering:
        subs_ss[v] = 0
        subs_ss[v(1)] = 0
    for s in shocks:
        subs_ss[s] = 0

    res = sympy.Matrix([eq.set_eq_zero for eq in eqs])

    F1_sym = res.jacobian(xp + yp).subs(subs_ss)
    F0_sym = res.jacobian(x0 + y0).subs(subs_ss)
    Fu_sym = res.jacobian(shocks).subs(subs_ss)

    eval_F1 = model.lambdify(F1_sym, context=context_f)
    eval_F0 = model.lambdify(F0_sym, context=context_f)
    eval_Fu = model.lambdify(Fu_sym, context=context_f)

    # Hessian tensor wrt v = [x', y', x, y, u]
    v = xp + yp + x0 + y0 + shocks
    nv = len(v)
    tensor_entries = n * nv * nv

    eval_Hi = []
    for i in range(n):
        Hi_sym = sympy.hessian(res[i], v).subs(subs_ss)
        eval_Hi.append(model.lambdify(Hi_sym, context=context_f))

    eval_QQ = model.lambdify(model["covariance"], context=context_f)

    return _SecondOrderCompiled(
        x0=x0,
        y0=y0,
        shocks=shocks,
        nx=nx,
        ny=ny,
        nu=nu,
        n=n,
        tensor_entries=tensor_entries,
        v=v,
        eval_F1=eval_F1,
        eval_F0=eval_F0,
        eval_Fu=eval_Fu,
        eval_Hi=eval_Hi,
        eval_QQ=eval_QQ,
    )


def _get_second_order_compiled(model) -> _SecondOrderCompiled:
    compiled = getattr(model, "_second_order_compiled", None)
    if compiled is None:
        compiled = _compile_second_order(model)
        setattr(model, "_second_order_compiled", compiled)
    return compiled


def _eliminate_lagged_endogenous(
    var_ordering: Sequence[Variable],
    equations: Sequence[Equation],
) -> Tuple[List[Variable], List[Equation], List[Variable]]:
    """
    Returns (new_var_ordering, new_equations, state_vars) where:
    - all occurrences of v(-1) in equations are replaced by an auxiliary state var v_LAG1
    - a linking equation v_LAG1(1) = v is added for each introduced state var
    """
    existing_names = [v.name for v in var_ordering]

    # Identify lagged endogenous atoms (date == -1).
    lagged_atoms: Dict[str, Variable] = {}
    for eq in equations:
        for atom in eq.atoms(Variable):
            if atom.date == -1:
                lagged_atoms[atom.name] = Variable(atom.name)

    if not lagged_atoms:
        return list(var_ordering), list(equations), []

    lag_map: Dict[Variable, Variable] = {}
    aux_by_base_name: Dict[str, Variable] = {}
    state_vars: List[Variable] = []
    new_var_ordering = list(var_ordering)

    for base_name, base_var in sorted(lagged_atoms.items(), key=lambda kv: kv[0]):
        aux_name = _make_unique_name(existing_names, f"{base_name}_LAG1")
        existing_names.append(aux_name)
        aux_var = Variable(aux_name)
        aux_by_base_name[base_name] = aux_var
        state_vars.append(aux_var)
        new_var_ordering.append(aux_var)
        lag_map[Variable(base_name, date=-1)] = aux_var

    new_equations = [eq.subs(lag_map) for eq in equations]

    # Add linking equations: aux(+1) = base
    for base_name, base_var in sorted(lagged_atoms.items(), key=lambda kv: kv[0]):
        new_equations.append(Equation(aux_by_base_name[base_name](1), base_var))

    return new_var_ordering, new_equations, state_vars


def solve_second_order(
    model,
    p0: Sequence[float],
    *,
    max_tensor_entries: int = 2_000_000,
    bk_tol: float = 1e-8,
) -> SecondOrderSolution:
    """
    Computes a second-order perturbation solution for LRE DSGE models.

    Notes
    -----
    - This solver treats any endogenous lag v(-1) as a state by introducing an auxiliary
      variable v_LAG1 and adding the linking identity v_LAG1(+1) = v.
    - Only intended for standard LRE models (not SI/FHP/OBC).
    """
    compiled = _get_second_order_compiled(model)
    if compiled.tensor_entries > max_tensor_entries:
        raise ValueError(
            f"Second-order tensor too large ({compiled.tensor_entries} entries). "
            f"Reduce model size or increase max_tensor_entries."
        )

    x0 = compiled.x0
    y0 = compiled.y0
    shocks = compiled.shocks
    nx = compiled.nx
    ny = compiled.ny
    nu = compiled.nu
    n = compiled.n

    F1 = np.asarray(compiled.eval_F1(p0), dtype=float)
    F0 = np.asarray(compiled.eval_F0(p0), dtype=float)
    Fu = np.asarray(compiled.eval_Fu(p0), dtype=float)

    f1 = F1[:, :nx]
    f2 = F1[:, nx:]
    f3 = F0[:, :nx]
    f4 = F0[:, nx:]
    f5 = Fu

    A = np.concatenate([f1, f2], axis=1)  # (n, n) on s_{t+1}
    B = -np.concatenate([f3, f4], axis=1)  # (n, n) on s_t

    # Solve first-order using QZ on (B, A) to get eigenvalues of A^{-1} B.
    S, T, alpha, beta, Q, Z = ordqz(B, A, sort="iuc", output="complex")
    with np.errstate(divide="ignore", invalid="ignore"):
        eig = np.divide(alpha, beta)
    stable_count = int(np.sum(np.abs(eig) < (1.0 - bk_tol)))
    if stable_count != nx:
        raise ValueError(
            f"Blanchard-Kahn condition failed: expected {nx} stable roots, got {stable_count}."
        )

    Z11 = Z[:nx, :nx]
    Z21 = Z[nx:, :nx]
    if np.linalg.matrix_rank(Z11) < nx:
        raise ValueError("Singular stable subspace (Z11 not full rank).")

    gx = np.real_if_close(Z21 @ np.linalg.inv(Z11))
    T11 = T[:nx, :nx]
    S11 = S[:nx, :nx]
    hx = np.real_if_close(Z11 @ np.linalg.inv(T11) @ S11 @ np.linalg.inv(Z11))

    Mx = f1 + f2 @ gx
    M = np.concatenate([Mx, f4], axis=1)
    if np.linalg.matrix_rank(M) < n:
        raise ValueError("Singular linear system when solving shock impact matrices.")

    X = np.linalg.solve(M, -f5)
    hu = X[:nx, :]
    gu = X[nx:, :]

    nv = len(compiled.v)
    H = np.zeros((n, nv, nv), dtype=float)
    for i, eval_hi in enumerate(compiled.eval_Hi):
        H[i] = np.asarray(eval_hi(p0), dtype=float)

    # Build derivative maps v_x and v_u
    v_x = np.vstack(
        [
            hx,  # x' wrt x
            gx @ hx,  # y' wrt x
            np.eye(nx),  # x wrt x
            gx,  # y wrt x
            np.zeros((nu, nx)),  # u wrt x
        ]
    )
    v_u = np.vstack(
        [
            hu,  # x' wrt u
            gx @ hu,  # y' wrt u (through x')
            np.zeros((nx, nu)),  # x wrt u
            gu,  # y wrt u
            np.eye(nu),  # u wrt u
        ]
    )

    Rxx = np.einsum("rab,ai,bj->rij", H, v_x, v_x)
    Rxu = np.einsum("rab,ai,bj->rij", H, v_x, v_u)
    Ruu = np.einsum("rab,ai,bj->rij", H, v_u, v_u)

    # Solve coupled system for (hxx, gxx)
    K = np.kron(hx, hx)
    Rxx_mat = Rxx.reshape(n, nx * nx, order="F")
    b_xx = -Rxx_mat.flatten(order="F")

    Ah = np.kron(np.eye(nx * nx), Mx)
    Ag = np.kron(np.eye(nx * nx), f4) + np.kron(K.T, f2)
    A_xx = np.concatenate([Ah, Ag], axis=1)
    sol_xx = np.linalg.solve(A_xx, b_xx)

    hxx_mat = sol_xx[: nx * nx * nx].reshape(nx, nx * nx, order="F")
    gxx_mat = sol_xx[nx * nx * nx :].reshape(ny, nx * nx, order="F")

    hxx = np.zeros((nx, nx, nx), dtype=float)
    for i in range(nx):
        hxx[i] = hxx_mat[i].reshape(nx, nx, order="F")
    gxx = np.zeros((ny, nx, nx), dtype=float)
    for i in range(ny):
        gxx[i] = gxx_mat[i].reshape(nx, nx, order="F")

    # Symmetrize in (x,x)
    hxx = 0.5 * (hxx + np.swapaxes(hxx, 1, 2))
    gxx = 0.5 * (gxx + np.swapaxes(gxx, 1, 2))

    # Solve (hxu, gxu) columnwise
    Rxu_mat = Rxu.reshape(n, nx * nu, order="F")
    ypxu_known = np.einsum("kpq,pi,qj->kij", gxx, hx, hu).reshape(ny, nx * nu, order="F")
    b_xu = -(Rxu_mat + f2 @ ypxu_known)
    sol_xu = np.linalg.solve(M, b_xu)
    hxu_mat = sol_xu[:nx, :]
    gxu_mat = sol_xu[nx:, :]

    hxu = np.zeros((nx, nx, nu), dtype=float)
    for i in range(nx):
        hxu[i] = hxu_mat[i].reshape(nx, nu, order="F")
    gxu = np.zeros((ny, nx, nu), dtype=float)
    for i in range(ny):
        gxu[i] = gxu_mat[i].reshape(nx, nu, order="F")

    # Solve (huu, guu) columnwise
    Ruu_mat = Ruu.reshape(n, nu * nu, order="F")
    ypuu_known = np.einsum("kpq,pi,qj->kij", gxx, hu, hu).reshape(ny, nu * nu, order="F")
    b_uu = -(Ruu_mat + f2 @ ypuu_known)
    sol_uu = np.linalg.solve(M, b_uu)
    huu_mat = sol_uu[:nx, :]
    guu_mat = sol_uu[nx:, :]

    huu = np.zeros((nx, nu, nu), dtype=float)
    for i in range(nx):
        huu[i] = huu_mat[i].reshape(nu, nu, order="F")
    guu = np.zeros((ny, nu, nu), dtype=float)
    for i in range(ny):
        guu[i] = guu_mat[i].reshape(nu, nu, order="F")

    # Symmetrize in (u,u)
    huu = 0.5 * (huu + np.swapaxes(huu, 1, 2))
    guu = 0.5 * (guu + np.swapaxes(guu, 1, 2))

    # Constant (risk) terms using shock covariance
    QQ = np.asarray(compiled.eval_QQ(p0), dtype=float)
    if QQ.shape != (nu, nu):
        raise ValueError(f"Shock covariance has shape {QQ.shape}, expected {(nu, nu)}.")

    gsig = np.einsum("kij,ij->k", guu, QQ)
    V = gu @ QQ @ gu.T
    yprime_start = nx
    yprime_end = nx + ny
    H_yppy = H[:, yprime_start:yprime_end, yprime_start:yprime_end]
    c_var = np.einsum("rab,ab->r", H_yppy, V)

    M_const = np.concatenate([Mx, (f2 + f4)], axis=1)
    rhs_const = -(f2 @ gsig + c_var)
    sol_const = np.linalg.solve(M_const, rhs_const)
    hss = np.asarray(sol_const[:nx], dtype=float)
    gss = np.asarray(sol_const[nx:], dtype=float)

    return SecondOrderSolution(
        state_names=[v.name for v in x0],
        control_names=[v.name for v in y0],
        shock_names=[s.name for s in shocks],
        hx=np.asarray(hx, dtype=float),
        gx=np.asarray(gx, dtype=float),
        hu=np.asarray(hu, dtype=float),
        gu=np.asarray(gu, dtype=float),
        hxx=hxx,
        gxx=gxx,
        hxu=hxu,
        gxu=gxu,
        huu=huu,
        guu=guu,
        hss=hss,
        gss=gss,
        shock_cov=QQ,
    )
