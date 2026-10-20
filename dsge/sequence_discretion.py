"""Linear-quadratic discretionary policy from anticipated impulse responses.

The recursion follows Hebden and Winkler (2024), equations (24)--(25).
Unlike a sequence commitment problem, each current policymaker is constrained
by the response rules of subsequent policymakers, including off-path responses.
All arrays in this module use time-major ordering for outcomes and instruments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve, svdvals


class DiscretionError(ValueError):
    """The discretionary policy equations lack a unique numerical solution."""


class DiscretionConvergenceError(DiscretionError):
    """The stationary discretionary-policy iteration did not converge."""


@dataclass(frozen=True)
class DiscretionStep:
    """Current first-order condition and its admissible deviation response."""

    operator: np.ndarray
    offset: np.ndarray
    effective_response: np.ndarray
    residual: float


@dataclass(frozen=True)
class DiscretionResult:
    """Policy restrictions ``operator @ y + offset = 0`` and diagnostics.

    Rows are normalized without changing their zero sets. ``residual`` is the
    relative fixed-point residual for ``stationary`` and the relative residual
    of the future-policy tangent restrictions for ``finite``. No baseline is
    required to construct the policy operator.
    """

    operator: np.ndarray
    offset: np.ndarray
    effective_response: np.ndarray
    iterations: int
    residual: float
    method: str
    converged: bool = True

    @property
    def D(self) -> np.ndarray:
        """Alias for the policy operator."""
        return self.operator

    @property
    def a(self) -> np.ndarray:
        """Alias for the affine policy offset."""
        return self.offset

    @property
    def G(self) -> np.ndarray:
        """Alias for the current deviation's effective response."""
        return self.effective_response


def _finite_array(value, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array.")
    return array


def _solve_unique(a: np.ndarray, b: np.ndarray, label: str,
                  rank_tol: float) -> np.ndarray:
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{label} must be square.")
    if a.shape[0] == 0:
        return np.zeros_like(b)
    singular_values = svdvals(a, check_finite=False)
    if (singular_values[0] == 0 or
            singular_values[-1] <= rank_tol * singular_values[0]):
        raise DiscretionError(
            f"{label} is singular or numerically rank deficient; "
            "the policy response is not uniquely identified."
        )
    try:
        return solve(a, b, assume_a="gen", check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise DiscretionError(f"{label} could not be solved.") from exc


def _normalize(operator: np.ndarray,
               offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Scaling the affine row as a whole preserves its exact zero set. Positive
    # row scales also preserve multiplier signs in callers' constraint systems.
    scales = np.linalg.norm(operator, axis=1)
    if np.any(scales == 0) or not np.all(np.isfinite(scales)):
        raise DiscretionError("A policy restriction has a zero or nonfinite row.")
    return operator / scales[:, None], offset / scales


def discretionary_step(
    M: np.ndarray,
    Q: np.ndarray,
    future_operator: np.ndarray,
    gradient: np.ndarray | None = None,
    *,
    rank_tol: float = 1e-12,
) -> DiscretionStep:
    """Compute one discretionary first-order condition.

    ``M`` maps all current and anticipated future instrument innovations to the
    outcome path. ``Q`` and ``gradient`` define its objective
    ``0.5 * y.T @ Q @ y + gradient.T @ y``. The future policymakers require
    ``future_operator @ y + c = 0``; the offset ``c`` is retained by the caller
    when stacking policy restrictions, but does not affect the tangent below.

    If there are ``k`` instrument columns and ``r`` future restrictions, the
    first ``k-r`` columns are the current instruments. With ``M=[M0,Mf]`` and
    ``C=future_operator``, this computes

    ``G = M0 - Mf @ solve(C @ Mf, C @ M0)``.

    Thus ``C @ G = 0``: a current deviation incorporates future policymakers'
    reactions. Its first-order condition is ``G.T @ (Q @ y + gradient)=0``.
    There must be a unique future response and positive definite loss
    curvature along current admissible deviations. No pseudoinverse or ridge
    is substituted for a missing equilibrium condition.
    """
    M = _finite_array(M, "M", 2)
    Q = _finite_array(Q, "Q", 2)
    C = _finite_array(future_operator, "future_operator", 2)
    if not 0 < rank_tol < 1:
        raise ValueError("rank_tol must be strictly between zero and one.")
    rows, columns = M.shape
    if Q.shape != (rows, rows) or C.shape[1] != rows:
        raise ValueError("Q and future_operator must match the outcome dimension.")
    current = columns - C.shape[0]
    if current <= 0:
        raise ValueError("There must be at least one current instrument column.")
    g = (np.zeros(rows) if gradient is None
         else _finite_array(gradient, "gradient", 1))
    if g.shape != (rows,):
        raise ValueError("gradient must have one entry per outcome-path element.")
    if not np.allclose(Q, Q.T, atol=1e-12, rtol=1e-10):
        raise ValueError("Q must be symmetric.")
    M0, Mf = M[:, :current], M[:, current:]
    if C.shape[0]:
        reaction = _solve_unique(C @ Mf, C @ M0,
                                 "Future-policy response matrix", rank_tol)
        G = M0 - Mf @ reaction
    else:
        G = M0.copy()
    curvature = G.T @ Q @ G
    curvature = 0.5 * (curvature + curvature.T)
    eigenvalues = np.linalg.eigvalsh(curvature)
    if (eigenvalues[-1] <= 0 or
            eigenvalues[0] <= rank_tol * eigenvalues[-1]):
        raise DiscretionError(
            "The current policy loss does not have positive definite curvature "
            "along admissible deviations."
        )
    tangent_residual = float(
        np.linalg.norm(C @ G) /
        max(1.0, np.linalg.norm(C) * np.linalg.norm(G))
    )
    return DiscretionStep(G.T @ Q, G.T @ g, G, tangent_residual)


def shift_policy_operator(operator: np.ndarray, n_variables: int,
                          terminal: str = "repeat") -> np.ndarray:
    """Right-multiply policy rows by the finite forward-shift operator.

    The paper's ``repeat`` closure repeats the last available outcome block;
    ``zero`` appends a zero block. These are numerical terminal closures, not
    proofs that a finite truncation equals its infinite-horizon limit.
    """
    operator = _finite_array(operator, "operator", 2)
    if (not isinstance(n_variables, (int, np.integer)) or n_variables < 1 or
            operator.shape[1] % n_variables):
        raise ValueError("n_variables must divide the operator's column count.")
    if terminal not in {"repeat", "zero"}:
        raise ValueError("terminal must be 'repeat' or 'zero'.")
    shifted = np.zeros_like(operator)
    shifted[:, n_variables:] = operator[:, :-n_variables]
    if terminal == "repeat":
        shifted[:, -n_variables:] += operator[:, -n_variables:]
    return shifted


def solve_discretion(
    M: np.ndarray,
    W: np.ndarray,
    discount: float,
    linear_loss: np.ndarray | None = None,
    method: str = "stationary",
    tol: float = 1e-9,
    max_iter: int = 1000,
    damping: float = 1.0,
    *,
    terminal: str = "repeat",
    rank_tol: float = 1e-12,
) -> DiscretionResult:
    """Construct finite or stationary discretionary policy restrictions.

    Parameters
    ----------
    M
        Anticipated-response matrix of shape ``(N*n, N*m)`` with both outcomes
        and instrument columns in **time-major** order. Its leading submatrices
        must describe fresh-date responses at shorter horizons, as for a
        time-invariant linear model; bottom-right news-response submatrices do
        not generally have that interpretation.
    W, linear_loss
        Symmetric positive semidefinite per-period Hessian and optional gradient
        for ``0.5*y_t.T@W@y_t + linear_loss.T@y_t``. Instrument penalties belong
        in these outcome variables. Constants in the objective are irrelevant.
    discount
        Per-period discount in ``(0, 1]``.
    method
        ``finite`` performs backward induction with losses ending at date N-1
        and the terminal private-sector continuation encoded by ``M``. It is
        exact for that truncated game. ``stationary`` iterates equations
        (24)--(25) of Hebden--Winkler using a fixed N-date truncation and the
        selected terminal shift. Convergence is checked, not assumed.
    damping
        Fraction of the new row-normalized operator used per stationary update.
    terminal
        ``repeat`` implements the paper's finite shift; ``zero`` offers a
        zero-tail closure. It is used only by the stationary method.

    Returns
    -------
    DiscretionResult
        For a baseline ``b``, solve ``(D @ M) u = -(D @ b + a)`` and set
        ``y=b+M@u``. ``D`` is row-equivalent to the paper's operator. Normalizing
        each row avoids multiplying successive future conditions by discount
        powers that would otherwise underflow; their solution set is unchanged.

    Notes
    -----
    This routine computes discretionary restrictions, not a rolling commitment
    optimization. It assumes valid anticipated responses, a unique continuation
    response at each step, and strict curvature in the current controls. A
    stationary result is a finite approximation and should be checked as the
    response horizon grows. No model-uncertainty or occasionally-binding-
    constraint assumptions are added by this routine.
    """
    M = _finite_array(M, "M", 2)
    W = _finite_array(W, "W", 2)
    if W.shape[0] == 0 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a nonempty square matrix.")
    n = W.shape[0]
    if M.shape[0] == 0 or M.shape[0] % n:
        raise ValueError("M must have N*n outcome rows for some positive N.")
    N = M.shape[0] // n
    if M.shape[1] == 0 or M.shape[1] % N:
        raise ValueError("M must have N*m instrument columns for some positive m.")
    m = M.shape[1] // N
    if not np.allclose(W, W.T, atol=1e-12, rtol=1e-10):
        raise ValueError("W must be symmetric.")
    W = 0.5 * (W + W.T)
    eigenvalues = np.linalg.eigvalsh(W)
    if eigenvalues[0] < -1e-12 * max(1.0, abs(eigenvalues[-1])):
        raise ValueError("W must be positive semidefinite.")
    if not np.isfinite(discount) or not 0 < discount <= 1:
        raise ValueError("discount must be in (0, 1].")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(damping) or not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1].")
    if not np.isfinite(rank_tol) or not 0 < rank_tol < 1:
        raise ValueError("rank_tol must be strictly between zero and one.")
    if method not in {"finite", "stationary"}:
        raise ValueError("method must be 'finite' or 'stationary'.")
    if terminal not in {"repeat", "zero"}:
        raise ValueError("terminal must be 'repeat' or 'zero'.")
    g = (np.zeros(n) if linear_loss is None
         else _finite_array(linear_loss, "linear_loss", 1))
    if g.shape != (n,):
        raise ValueError("linear_loss must have one entry per per-period variable.")
    weights = discount ** np.arange(N)
    if np.any(weights == 0):
        raise ValueError("Discount powers underflow at the requested horizon.")
    Q = np.kron(np.diag(weights), W)
    gradient = np.kron(weights, g)

    if method == "finite":
        D = np.empty((0, 0))
        a = np.empty(0)
        for length in range(1, N + 1):
            C = np.column_stack((np.zeros((D.shape[0], n)), D))
            current_M = M[:length*n, :length*m]
            step = discretionary_step(
                current_M, Q[:length*n, :length*n], C,
                gradient[:length*n], rank_tol=rank_tol,
            )
            # The beta multiplier in eq. (25) is an irrelevant positive row
            # scale; omit it before normalization to avoid tail underflow.
            D, a = _normalize(
                np.vstack((step.operator, C)),
                np.concatenate((step.offset, a)),
            )
        _solve_unique(D @ M, np.zeros(N*m), "Policy system", rank_tol)
        return DiscretionResult(D, a, step.effective_response, N,
                                step.residual, method)

    D, a = _normalize(M.T @ Q, M.T @ gradient)
    residual = np.inf
    for iteration in range(1, max_iter + 1):
        C = shift_policy_operator(D[:-m], n, terminal)
        step = discretionary_step(M, Q, C, gradient, rank_tol=rank_tol)
        proposed_D, proposed_a = _normalize(
            np.vstack((step.operator, C)),
            np.concatenate((step.offset, a[:-m])),
        )
        # Check the operator separately: a large affine target must not mask
        # an unconverged reaction rule in another variable or instrument.
        operator_residual = (np.linalg.norm(proposed_D-D) /
                             max(1.0, np.linalg.norm(D)))
        offset_residual = np.max(
            np.abs(proposed_a-a) / np.maximum(1.0, np.abs(a)), initial=0.0,
        )
        residual = float(max(operator_residual, offset_residual))
        if residual <= tol:
            _solve_unique(D @ M, np.zeros(N*m), "Policy system", rank_tol)
            return DiscretionResult(D, a, step.effective_response, iteration,
                                    residual, method)
        D, a = _normalize(
            (1-damping)*D + damping*proposed_D,
            (1-damping)*a + damping*proposed_a,
        )
    raise DiscretionConvergenceError(
        f"Stationary discretion did not converge in {max_iter} iterations "
        f"(relative fixed-point residual {residual:.3g}). Increase max_iter, "
        "adjust damping, or inspect the response horizon and equilibrium."
    )
