from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

try:  # SciPy is already a dependency of dsge, but keep a small fallback.
    from scipy.linalg import qr as _scipy_qr
except Exception:  # pragma: no cover
    _scipy_qr = None  # type: ignore[assignment]


ReductionMode = Literal["minimal", "observable", "controllable"]


@dataclass(frozen=True)
class ReductionInfo:
    mode: str
    tol: float
    max_steps: int
    ns_original: int
    ns_final: int
    ns_controllable: int
    ns_observable: int
    dropped_controllable: bool
    dropped_observable: bool
    skipped_controllable_reason: Optional[str] = None


def _as_float_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _colspace_basis(mat: np.ndarray, *, tol: float) -> np.ndarray:
    mat = _as_float_array(mat)
    n, m = mat.shape
    if n == 0 or m == 0:
        return np.zeros((n, 0), dtype=float)

    if _scipy_qr is not None:
        q, r, _ = _scipy_qr(mat, mode="economic", pivoting=True)
        diag = np.abs(np.diag(r))
        if diag.size == 0:
            return np.zeros((n, 0), dtype=float)
        s0 = float(diag[0])
        if not np.isfinite(s0) or s0 == 0.0:
            return np.zeros((n, 0), dtype=float)
        cutoff = float(tol) * s0
        rank = int(np.sum(diag > cutoff))
        return np.ascontiguousarray(q[:, :rank])

    # Fallback: SVD (no pivoting).
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    if s.size == 0:
        return np.zeros((n, 0), dtype=float)
    s0 = float(s[0])
    if not np.isfinite(s0) or s0 == 0.0:
        return np.zeros((n, 0), dtype=float)
    cutoff = float(tol) * s0
    rank = int(np.sum(s > cutoff))
    return np.ascontiguousarray(u[:, :rank])


def _reachable_subspace_basis(
    a: np.ndarray,
    b: np.ndarray,
    *,
    tol: float,
    max_steps: int,
) -> np.ndarray:
    a = _as_float_array(a)
    b = _as_float_array(b)
    n = int(a.shape[0])

    q = _colspace_basis(b, tol=tol)
    if q.shape[1] == 0:
        return q

    max_steps = int(max(1, min(max_steps, n)))
    for _ in range(max_steps - 1):
        k_old = int(q.shape[1])
        aq = a @ q
        q = _colspace_basis(np.concatenate([q, aq], axis=1), tol=tol)
        if int(q.shape[1]) == k_old or int(q.shape[1]) == n:
            break
    return q


def _observable_subspace_basis(
    a: np.ndarray,
    c: np.ndarray,
    *,
    tol: float,
    max_steps: int,
) -> np.ndarray:
    # Observable subspace of (A, C) == reachable subspace of (A.T, C.T).
    return _reachable_subspace_basis(a.T, c.T, tol=tol, max_steps=max_steps)


def _sqrt_psd_factor(q: np.ndarray, *, tol: float) -> np.ndarray:
    q = _as_float_array(q)
    if q.size == 0:
        return q.reshape(q.shape[0], 0)

    q = 0.5 * (q + q.T)
    eigvals, eigvecs = np.linalg.eigh(q)
    # Clamp tiny negative eigenvalues (numerical) to zero.
    eigvals = np.where(eigvals > 0.0, eigvals, 0.0)
    s0 = float(np.max(eigvals)) if eigvals.size else 0.0
    cutoff = float(tol) * max(1.0, s0)
    keep = eigvals > cutoff
    if not np.any(keep):
        return np.zeros((q.shape[0], 0), dtype=float)
    return eigvecs[:, keep] * np.sqrt(eigvals[keep])[None, :]


def _proj_residual_norm(v: np.ndarray, q: np.ndarray) -> float:
    v = _as_float_array(v).reshape(-1)
    if q.size == 0:
        return float(np.linalg.norm(v))
    proj = q @ (q.T @ v)
    return float(np.linalg.norm(v - proj))


def _matrix_proj_residual_fro(m: np.ndarray, q: np.ndarray) -> float:
    m = _as_float_array(m)
    if q.size == 0:
        return float(np.linalg.norm(m))
    mq = m @ q
    proj = q @ (q.T @ mq) @ q.T
    return float(np.linalg.norm(m - proj))


def reduce_state_space(
    CC: np.ndarray,
    TT: np.ndarray,
    RR: np.ndarray,
    QQ: np.ndarray,
    DD: np.ndarray,
    ZZ: np.ndarray,
    HH: np.ndarray,
    A0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    *,
    mode: ReductionMode = "minimal",
    tol: float = 1e-10,
    max_steps: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], ReductionInfo]:
    """
    Exact (rank-based) state-space reduction.

    This performs an optional Kalman-style decomposition:
    - drop uncontrollable (unreachable) states w.r.t. stochastic shocks (RR, QQ),
      but only when doing so is safe given (CC, A0, P0).
    - drop unobservable states w.r.t. (TT, ZZ) (always safe).

    Notes
    -----
    - The reduction is conservative: it only removes directions that are numerically
      rank-deficient given `tol`.
    - For DSGE linear state-space systems, CC is typically zero and P0 is typically
      the unconditional covariance; in that common case, dropping uncontrollable
      states is safe and can speed up filtering.
    """
    mode_str = str(mode).lower().strip()
    if mode_str not in {"minimal", "observable", "controllable"}:
        raise ValueError(f"mode must be one of {{'minimal','observable','controllable'}}, got {mode!r}.")
    mode = mode_str  # type: ignore[assignment]

    tol = float(tol)
    if tol <= 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be a positive finite float.")

    CC = _as_float_array(CC).reshape(-1)
    TT = _as_float_array(TT)
    RR = _as_float_array(RR)
    QQ = _as_float_array(QQ)
    DD = _as_float_array(DD).reshape(-1)
    ZZ = _as_float_array(ZZ)
    HH = _as_float_array(HH)

    if A0 is not None:
        A0 = _as_float_array(A0).reshape(-1)
    if P0 is not None:
        P0 = _as_float_array(P0)

    ns_original = int(TT.shape[0])
    ns = ns_original
    max_steps_i = int(ns if max_steps is None else max(1, min(int(max_steps), ns)))

    ns_cont = ns
    ns_obs = ns
    dropped_controllable = False
    dropped_observable = False
    skipped_controllable_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # 1) Controllability reduction (optional + guarded)
    # ------------------------------------------------------------------
    if mode in {"minimal", "controllable"}:
        # Effective noise directions: RR * sqrt(QQ).
        sqrt_QQ = _sqrt_psd_factor(QQ, tol=tol)
        b = RR @ sqrt_QQ
        qc = _reachable_subspace_basis(TT, b, tol=tol, max_steps=max_steps_i)
        ns_cont = int(qc.shape[1])

        # qc == empty means "no stochastic input"; do not drop states based on that alone.
        if ns_cont == 0:
            skipped_controllable_reason = "no_stochastic_shocks"
        elif ns_cont < ns:
            # Only safe to drop if the discarded directions carry no mean/cov mass.
            cc_res = _proj_residual_norm(CC, qc)
            a0_res = 0.0 if A0 is None else _proj_residual_norm(A0, qc)
            p0_res = 0.0 if P0 is None else _matrix_proj_residual_fro(P0, qc)

            scale = max(1.0, float(np.linalg.norm(CC)))
            if A0 is not None:
                scale = max(scale, float(np.linalg.norm(A0)))
            if P0 is not None:
                scale = max(scale, float(np.linalg.norm(P0)))

            if max(cc_res, a0_res, p0_res) > tol * scale:
                skipped_controllable_reason = "nonzero_initial_outside_controllable_subspace"
            else:
                # Project into controllable subspace.
                CC = qc.T @ CC
                TT = qc.T @ TT @ qc
                RR = qc.T @ RR
                ZZ = ZZ @ qc
                if A0 is not None:
                    A0 = qc.T @ A0
                if P0 is not None:
                    P0 = qc.T @ P0 @ qc
                dropped_controllable = True
                ns = ns_cont

    # ------------------------------------------------------------------
    # 2) Observability reduction (always safe)
    # ------------------------------------------------------------------
    if mode in {"minimal", "observable"}:
        qo = _observable_subspace_basis(TT, ZZ, tol=tol, max_steps=max_steps_i)
        ns_obs = int(qo.shape[1])
        if ns_obs < ns:
            CC = qo.T @ CC
            TT = qo.T @ TT @ qo
            RR = qo.T @ RR
            ZZ = ZZ @ qo
            if A0 is not None:
                A0 = qo.T @ A0
            if P0 is not None:
                P0 = qo.T @ P0 @ qo
            dropped_observable = True
            ns = ns_obs

    # Ensure consistent dtypes/contiguity for downstream (Numba).
    CC = np.ascontiguousarray(CC, dtype=float)
    TT = np.ascontiguousarray(TT, dtype=float)
    RR = np.ascontiguousarray(RR, dtype=float)
    QQ = np.ascontiguousarray(QQ, dtype=float)
    DD = np.ascontiguousarray(DD, dtype=float)
    ZZ = np.ascontiguousarray(ZZ, dtype=float)
    HH = np.ascontiguousarray(HH, dtype=float)
    if A0 is not None:
        A0 = np.ascontiguousarray(A0, dtype=float)
    if P0 is not None:
        P0 = np.ascontiguousarray(P0, dtype=float)

    info = ReductionInfo(
        mode=str(mode),
        tol=float(tol),
        max_steps=int(max_steps_i),
        ns_original=int(ns_original),
        ns_final=int(TT.shape[0]),
        ns_controllable=int(ns_cont),
        ns_observable=int(ns_obs),
        dropped_controllable=bool(dropped_controllable),
        dropped_observable=bool(dropped_observable),
        skipped_controllable_reason=skipped_controllable_reason,
    )

    return CC, TT, RR, QQ, DD, ZZ, HH, A0, P0, info
