"""Numerical helpers for Hebden--Winkler sequence-space policy problems.

All arrays in this module are time-major. These routines solve deterministic
paths from anticipated responses; no structural states are needed.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize


@dataclass(frozen=True)
class QuadraticSolution:
    choice: np.ndarray
    multipliers: np.ndarray
    slack: np.ndarray
    iterations: int
    residual: float


def checked_solve(matrix, rhs, *, tolerance=1e-9, name="policy system"):
    """Solve a consistent system, permitting redundant coordinates, never bad FOCs."""
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    try:
        result = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        result = np.linalg.lstsq(matrix, rhs, rcond=1e-12)[0]
    error = np.max(np.abs(matrix @ result - rhs), initial=0.0)
    if not np.all(np.isfinite(result)) or error > tolerance * (1 + np.max(np.abs(rhs), initial=0)):
        raise ValueError(f"{name} is singular/inconsistent (residual {error:.3g}).")
    return result


def solve_quadratic(H, f, A=None, b=None, *, tolerance=1e-9, max_iter=1000):
    """Convex quadratic program: min 1/2 u'Hu+f'u subject to A u >= b.

    The active-set solution is verified against primal feasibility, dual
    feasibility, stationarity and complementarity. A generic convex optimizer
    is used only to obtain a new active-set candidate if pivots cycle.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    H, f = np.asarray(H, float), np.asarray(f, float)
    if H.shape != (f.size, f.size) or f.ndim != 1:
        raise ValueError("Quadratic objective dimensions do not match.")
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(f)):
        raise ValueError("Quadratic objective coefficients must be finite.")
    H = (H + H.T) / 2
    scale = max(1.0, np.linalg.norm(H, ord=2))
    if np.linalg.eigvalsh(H).min(initial=0) < -tolerance * scale:
        raise ValueError("The policy objective must be convex.")
    A = np.empty((0, len(f))) if A is None else np.asarray(A, float)
    b = np.empty(0) if b is None else np.asarray(b, float)
    if A.shape != (len(b), len(f)):
        raise ValueError("Constraint dimensions do not match the controls.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("Constraint coefficients must be finite.")

    def optimizer_candidate(start):
        # A constrained optimum can exist even if the unconstrained quadratic
        # has no stationary point (e.g. minimize u subject to u >= 0).
        candidate = minimize(
            lambda x: 0.5 * x @ H @ x + f @ x, start,
            jac=lambda x: H @ x + f, method="SLSQP",
            constraints=[{"type": "ineq", "fun": lambda x: A @ x - b,
                          "jac": lambda x: A}],
            options={"ftol": min(tolerance, 1e-12), "maxiter": max_iter},
        )
        if not candidate.success:
            raise ValueError(f"Constrained policy problem failed: {candidate.message}")
        return list(np.flatnonzero(A @ candidate.x - b < 10 * tolerance))

    active = []
    visited = set()
    used_fallback = False
    u = np.zeros_like(f)
    for iteration in range(1, max_iter + 1):
        key = tuple(sorted(active))
        if key in visited:
            if used_fallback:
                raise ValueError("Constraint active-set iteration did not converge.")
            active = optimizer_candidate(u)
            visited.clear()
            used_fallback = True
        visited.add(tuple(sorted(active)))
        Aa = A[active]
        K = np.block([[H, -Aa.T], [Aa, np.zeros((len(active), len(active)))]])
        try:
            solution = checked_solve(K, np.r_[-f, b[active]], tolerance=tolerance,
                                     name="constrained optimality system")
        except ValueError:
            if used_fallback or not len(b):
                raise
            active = optimizer_candidate(u)
            visited.clear()
            used_fallback = True
            continue
        u = solution[:len(f)]
        multipliers = np.zeros(len(b))
        multipliers[active] = solution[len(f):]
        slack = A @ u - b
        inactive = np.ones(len(b), dtype=bool)
        inactive[active] = False
        violations = np.where(inactive, slack, np.inf)
        if violations.min(initial=np.inf) < -tolerance:
            active.append(int(np.argmin(violations)))
            continue
        if multipliers.min(initial=0) < -tolerance:
            active.remove(int(np.argmin(multipliers)))
            continue
        stationarity = H @ u + f - A.T @ multipliers
        residual = max(np.max(np.abs(stationarity), initial=0),
                       np.max(np.maximum(-slack, 0), initial=0),
                       np.max(np.maximum(-multipliers, 0), initial=0),
                       np.max(np.abs(slack * multipliers), initial=0))
        if residual > tolerance * 100 * (1 + np.max(np.abs(f), initial=0)):
            raise ValueError(f"Policy KKT verification failed (residual {residual:.3g}).")
        return QuadraticSolution(u, multipliers, slack, iteration, float(residual))
    raise ValueError("Constraint active-set iteration exceeded max_iter.")



@dataclass(frozen=True)
class ConstrainedDiscretionSolution:
    choice: np.ndarray
    outcomes: np.ndarray
    multipliers: np.ndarray
    slack: np.ndarray
    active: np.ndarray
    iterations: int
    residual: float
    operator: np.ndarray
    offset: np.ndarray


def solve_constrained_discretion(M, baseline, W, discount, C, bounds, *,
                                 linear_loss=None, binding_horizon=None,
                                 tolerance=1e-9, max_iter=500, terminal="repeat",
                                 unconstrained=None, method="stationary"):
    """HW section 5.3: fixed-future-active-set discretionary equilibrium.

    This is a local active-set algorithm, not a global nonlinear game solver.
    ``stationary`` uses the paper's unconstrained continuation; constraints
    after ``binding_horizon`` must be slack. ``finite`` ends the loss at the
    final date and permits constraints to bind there. Future binding sets are
    held fixed when taking derivatives in either case. Increase the horizon
    to check the effect of either terminal approximation.
    """
    from .sequence_discretion import solve_discretion, discretionary_step
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    n = W.shape[0]
    N = M.shape[0] // n
    m = M.shape[1] // N
    q = C.shape[0]
    if method not in {"stationary", "finite"}:
        raise ValueError("method must be 'stationary' or 'finite'.")
    if method == "finite" and binding_horizon is not None:
        raise ValueError("binding_horizon applies only to stationary discretion.")
    horizon = N - 1 if binding_horizon is None else int(binding_horizon)
    if not 0 <= horizon < N:
        raise ValueError("binding_horizon must be between zero and T-1.")
    if method == "stationary" and unconstrained is None:
        unconstrained = solve_discretion(M, W, discount, linear_loss=linear_loss,
                                         method="stationary", tol=tolerance)
    weights = discount ** np.arange(N)
    if np.any(weights == 0):
        raise ValueError("Discount powers underflow at the requested horizon.")
    Q = np.kron(np.diag(weights), W)
    gradient = np.kron(weights, np.zeros(n) if linear_loss is None else linear_loss)
    F = np.zeros((N*n, N*n))
    F[:(N-1)*n, n:] = np.eye((N-1)*n)
    if terminal == "repeat":
        F[-n:, -n:] = np.eye(n)
    C0 = np.zeros((q, N*n))
    C0[:, :n] = C
    active = np.zeros((N, q), bool)
    visited = set()
    for iteration in range(1, max_iter+1):
        key = active.tobytes()
        if key in visited:
            raise ValueError("Constrained discretion active sets cycle; a regular local solution was not found.")
        visited.add(key)
        if method == "stationary":
            D = unconstrained.operator.copy()
            offset = unconstrained.offset.copy()
            multiplier_map = np.zeros((N*q, N*n))
            multiplier_constant = np.zeros(N*q)
        else:
            D = np.empty((0, 0))
            offset = np.empty(0)
            multiplier_map = np.empty((0, 0))
            multiplier_constant = np.empty(0)
        for h in range(horizon, -1, -1):
            if method == "stationary":
                length = N
                future = D[:(N-1)*m] @ F
                future_offset = offset[:(N-1)*m]
                future_mu = multiplier_map[:(N-1)*q] @ F
                future_mu_constant = multiplier_constant[:(N-1)*q]
            else:
                length = N - h
                # Fresh-date anticipated responses are leading submatrices,
                # not bottom-right blocks of the original announcement map.
                future = np.column_stack((np.zeros((D.shape[0], n)), D))
                future_offset = offset
                future_mu = np.column_stack((
                    np.zeros((multiplier_map.shape[0], n)), multiplier_map,
                ))
                future_mu_constant = multiplier_constant
            step = discretionary_step(
                M[:length*n, :length*m], Q[:length*n, :length*n], future,
                gradient[:length*n],
            )
            G = step.effective_response
            rows = np.flatnonzero(active[h])
            A = C0[rows, :length*n]
            E = A @ G
            if len(rows) > m or np.linalg.matrix_rank(E, tol=1e-11) < len(rows):
                raise ValueError("Dependent or excessive binding constraints in discretionary step.")
            if len(rows):
                Z = null_space(E)
                d = np.vstack([Z.T @ step.operator, A])
                a = np.r_[Z.T @ step.offset, -bounds[rows]]
                K = checked_solve(E @ E.T, E, name="discretionary multiplier system")
                mu_map = np.zeros((q, length*n))
                mu_constant = np.zeros(q)
                mu_map[rows] = K @ step.operator
                mu_constant[rows] = K @ step.offset
            else:
                d, a = step.operator, step.offset
                mu_map = np.zeros((q, length*n))
                mu_constant = np.zeros(q)
            # Row scaling does not change constraints and avoids beta underflow.
            scales = np.maximum(np.linalg.norm(d, axis=1), 1e-30)
            D = np.vstack([d/scales[:, None], future])
            offset = np.r_[a/scales, future_offset]
            multiplier_map = np.vstack([mu_map, future_mu])
            multiplier_constant = np.r_[mu_constant, future_mu_constant]
        choice = checked_solve(D @ M, -(D @ baseline + offset),
                               tolerance=tolerance, name="discretionary path")
        outcomes = baseline + M @ choice
        slack = outcomes.reshape(N, n) @ C.T - bounds
        multipliers = (multiplier_map @ outcomes + multiplier_constant).reshape(N, q)
        violation = np.where(~active, slack, np.inf)
        negative = np.where(active, multipliers, np.inf)
        if np.min(violation, initial=np.inf) < -tolerance:
            h, j = np.unravel_index(np.argmin(violation), violation.shape)
            if h > horizon:
                raise ValueError("Constraints bind beyond binding_horizon; increase the horizon.")
            active[h, j] = True
            continue
        if np.min(negative, initial=np.inf) < -tolerance:
            h, j = np.unravel_index(np.argmin(negative), negative.shape)
            active[h, j] = False
            continue
        residual = max(np.max(np.abs(D @ outcomes + offset), initial=0),
                       np.max(np.maximum(-slack, 0), initial=0),
                       np.max(np.maximum(-multipliers, 0), initial=0),
                       np.max(np.abs(slack*multipliers), initial=0))
        if residual > 100*tolerance*(1+np.max(np.abs(outcomes), initial=0)):
            raise ValueError("Constrained discretion failed equilibrium residual verification.")
        return ConstrainedDiscretionSolution(choice, outcomes, multipliers, slack,
                                              active, iteration, residual, D, offset)
    raise ValueError("Constrained discretion exceeded max_iter.")
