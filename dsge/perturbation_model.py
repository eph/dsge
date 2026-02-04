from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as p

from .logging_config import get_logger

logger = get_logger("dsge.perturbation")


def _as_2d_array(y):
    if isinstance(y, p.DataFrame):
        return y.values
    arr = np.asarray(y)
    if arr.ndim == 1:
        return np.swapaxes(np.atleast_2d(arr), 0, 1)
    return arr


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative_sum = np.cumsum(weights)
    idx = np.searchsorted(cumulative_sum, positions, side="right")
    return idx


def _logpdf_mvnormal(resid: np.ndarray, inv_cov: np.ndarray, logdet_cov: float) -> np.ndarray:
    """
    resid: (n_particles, n_obs)
    """
    quad = np.einsum("ni,ij,nj->n", resid, inv_cov, resid)
    n_obs = resid.shape[1]
    return -0.5 * (quad + logdet_cov + n_obs * np.log(2.0 * np.pi))


def _ensure_cov_matrix(hh, jitter: float = 1e-12) -> np.ndarray:
    hh = np.asarray(hh, dtype=float)
    if hh.ndim == 0:
        hh = np.array([[float(hh)]])
    elif hh.ndim == 1:
        hh = np.diag(hh)

    if hh.shape[0] != hh.shape[1]:
        raise ValueError(f"Measurement error covariance must be square, got {hh.shape}.")

    if np.allclose(hh, 0.0):
        hh = hh + jitter * np.eye(hh.shape[0])
    return hh


@dataclass
class _SecondOrderPolicy:
    hx: np.ndarray
    gx: np.ndarray
    hu: np.ndarray
    gu: np.ndarray
    hxx: np.ndarray
    gxx: np.ndarray
    hxu: np.ndarray
    gxu: np.ndarray
    huu: np.ndarray
    guu: np.ndarray
    hss: np.ndarray
    gss: np.ndarray
    shock_cov: np.ndarray
    control_names: list[str]
    state_names: list[str]
    shock_names: list[str]


def _policy_from_solution(sol) -> _SecondOrderPolicy:
    return _SecondOrderPolicy(
        hx=np.asarray(sol.hx, dtype=float),
        gx=np.asarray(sol.gx, dtype=float),
        hu=np.asarray(sol.hu, dtype=float),
        gu=np.asarray(sol.gu, dtype=float),
        hxx=np.asarray(sol.hxx, dtype=float),
        gxx=np.asarray(sol.gxx, dtype=float),
        hxu=np.asarray(sol.hxu, dtype=float),
        gxu=np.asarray(sol.gxu, dtype=float),
        huu=np.asarray(sol.huu, dtype=float),
        guu=np.asarray(sol.guu, dtype=float),
        hss=np.asarray(sol.hss, dtype=float),
        gss=np.asarray(sol.gss, dtype=float),
        shock_cov=np.asarray(sol.shock_cov, dtype=float),
        control_names=list(sol.control_names),
        state_names=list(sol.state_names),
        shock_names=list(sol.shock_names),
    )


class PerturbationDSGEModel:
    """
    Nonlinear (order-2) perturbation DSGE model wrapper with a particle-filter likelihood.

    This class is intended to match the high-level API of `LinearDSGEModel` where possible
    (e.g., `log_lik`, `log_post`, `impulse_response`, `simulate`), while using a pruned
    second-order decision rule and a bootstrap particle filter.
    """

    def __init__(
        self,
        *,
        dsge_model,
        yy,
        t0: int = 0,
        shock_names: Optional[Sequence[str]] = None,
        state_names: Optional[Sequence[str]] = None,
        obs_names: Optional[Sequence[str]] = None,
        prior=None,
        parameter_names=None,
        order: int = 2,
        pruning: bool = True,
    ):
        if order != 2:
            raise ValueError("PerturbationDSGEModel currently supports only order=2.")

        self.dsge_model = dsge_model
        self.yy = yy
        self.t0 = t0

        self.shock_names = list(shock_names) if shock_names is not None else None
        self.state_names = list(state_names) if state_names is not None else None
        self.obs_names = list(obs_names) if obs_names is not None else None
        self.parameter_names = parameter_names

        self.prior = prior
        self.pruning = bool(pruning)

        self._nendo = len(self.dsge_model["var_ordering"])

        self._cached_policy_para = None
        self._cached_policy = None
        self._cached_measurement_para = None
        self._cached_measurement = None  # (DD, ZZ, HH, QQ)

        self._validate_linear_observables()

    def _validate_linear_observables(self) -> None:
        # Enforce: observables depend only on current-period endogenous vars and are affine.
        if "observables" not in self.dsge_model:
            return
        endo_vars = list(self.dsge_model["var_ordering"])
        obs_eqs = self.dsge_model.get("obs_equations", {})
        offenders = []
        for obs in self.dsge_model["observables"]:
            expr = obs_eqs.get(obs.name, None)
            if expr is None:
                continue
            dated = [v for v in expr.atoms() if getattr(v, "date", 0) != 0]
            if dated:
                offenders.append((obs.name, "dated vars", [str(v) for v in dated]))
                continue
            try:
                poly = np  # sentinel
                import sympy

                poly = sympy.Poly(expr, *endo_vars, domain="EX")
                if poly.total_degree() > 1:
                    offenders.append((obs.name, "nonlinear", str(expr)))
            except Exception:
                offenders.append((obs.name, "non-polynomial", str(expr)))

        if offenders:
            msg = "Observable equations must be affine in current endogenous variables for order=2.\n"
            for name, kind, detail in offenders:
                msg += f"- {name}: {kind}: {detail}\n"
            raise ValueError(msg.rstrip())

    def _measurement_matrices(self, para, use_cache: bool = False):
        if use_cache and self._cached_measurement_para is not None and np.allclose(
            np.asarray(para, dtype=float),
            np.asarray(self._cached_measurement_para, dtype=float),
        ):
            return self._cached_measurement

        self.dsge_model.python_sims_matrices()

        DD_full = np.atleast_1d(self.dsge_model.DD(para)).astype(float)
        ZZ_full = np.atleast_2d(self.dsge_model.ZZ(para)).astype(float)

        DD = DD_full.reshape(-1)
        ZZ = ZZ_full[:, : self._nendo]

        HH = _ensure_cov_matrix(self.dsge_model.HH(para))
        QQ = np.atleast_2d(self.dsge_model.QQ(para)).astype(float)

        self._cached_measurement_para = np.asarray(para, dtype=float).copy()
        self._cached_measurement = (DD, ZZ, HH, QQ)
        return DD, ZZ, HH, QQ

    def _policy(self, para, use_cache: bool = False) -> _SecondOrderPolicy:
        if use_cache and self._cached_policy_para is not None and np.allclose(
            np.asarray(para, dtype=float),
            np.asarray(self._cached_policy_para, dtype=float),
        ):
            return self._cached_policy

        sol = self.dsge_model.solve_second_order(para)
        pol = _policy_from_solution(sol)

        self._cached_policy_para = np.asarray(para, dtype=float).copy()
        self._cached_policy = pol

        if self.state_names is None:
            self.state_names = pol.state_names
        if self.shock_names is None:
            self.shock_names = pol.shock_names
        return pol

    def log_pr(self, para, *args, **kwargs):
        try:
            return self.prior.logpdf(para)
        except Exception:
            return 0.0

    def log_post(self, para, *args, **kwargs):
        x = self.log_lik(para, *args, **kwargs) + self.log_pr(para)
        if np.isnan(x):
            x = -1000000000.0
        if x < -1000000000.0:
            x = -1000000000.0
        return x

    def log_lik(self, para, use_cache: bool = False, *args, **kwargs):
        """
        Particle filter likelihood for order-2 perturbation model.

        Extra kwargs
        ------------
        nparticles : int
        seed : int | None
        resample_threshold : float in (0,1]
        filter : str (must be 'particle_filter')
        """
        t0 = kwargs.pop("t0", self.t0)
        yy = _as_2d_array(kwargs.pop("y", self.yy))
        filt = kwargs.pop("filter", "particle_filter")
        if filt != "particle_filter":
            raise ValueError("Order-2 compile supports only filter='particle_filter'.")

        nparticles = int(kwargs.pop("nparticles", 2000))
        seed = kwargs.pop("seed", None)
        resample_threshold = float(kwargs.pop("resample_threshold", 0.5))
        if not (0.0 < resample_threshold <= 1.0):
            raise ValueError("resample_threshold must be in (0, 1].")

        rng = np.random.default_rng(seed)

        pol = self._policy(para, use_cache=use_cache)
        DD, ZZ, HH, QQ = self._measurement_matrices(para, use_cache=use_cache)

        nstate = pol.hx.shape[0]
        nshocks = QQ.shape[0]
        nobs = DD.size

        invHH = np.linalg.inv(HH)
        sign, logdet = np.linalg.slogdet(HH)
        if sign <= 0:
            raise ValueError("Measurement error covariance must be positive definite (or near).")

        # Initialize particles at steady state (deviations = 0), in pruned components.
        x1 = np.zeros((nparticles, nstate))
        x2 = np.zeros((nparticles, nstate))

        cholQQ = np.linalg.cholesky(QQ)

        loglik = 0.0

        for t in range(t0, yy.shape[0]):
            y_t = np.asarray(yy[t]).reshape(-1)
            mask = np.isfinite(y_t)

            # Draw shocks for this period.
            eps = rng.standard_normal(size=(nparticles, nshocks)) @ cholQQ.T

            # Current-period controls (pruned).
            y1 = x1 @ pol.gx.T + eps @ pol.gu.T

            y2 = pol.gss + x2 @ pol.gx.T
            if self.pruning:
                y2 = (
                    y2
                    + 0.5 * np.einsum("kij,ni,nj->nk", pol.gxx, x1, x1)
                    + np.einsum("kij,ni,nj->nk", pol.gxu, x1, eps)
                    + 0.5 * np.einsum("kij,ni,nj->nk", pol.guu, eps, eps)
                )
            y_curr = y1 + y2

            yhat = DD + y_curr @ ZZ.T

            # Propagate to next state (pruned).
            x1_next = x1 @ pol.hx.T + eps @ pol.hu.T

            x2_next = pol.hss + x2 @ pol.hx.T
            if self.pruning:
                x2_next = (
                    x2_next
                    + 0.5 * np.einsum("kij,ni,nj->nk", pol.hxx, x1, x1)
                    + np.einsum("kij,ni,nj->nk", pol.hxu, x1, eps)
                    + 0.5 * np.einsum("kij,ni,nj->nk", pol.huu, eps, eps)
                )

            if not np.any(mask):
                # Missing observation: just move on.
                x1, x2 = x1_next, x2_next
                continue

            resid = y_t[mask][None, :] - yhat[:, mask]
            invHH_t = invHH[np.ix_(mask, mask)]
            logdet_t = np.linalg.slogdet(HH[np.ix_(mask, mask)])[1]
            logw = _logpdf_mvnormal(resid, invHH_t, logdet_t)

            maxlogw = float(np.max(logw))
            w = np.exp(logw - maxlogw)
            meanw = float(np.mean(w))
            if not np.isfinite(meanw) or meanw <= 0.0:
                return -np.inf
            loglik += maxlogw + np.log(meanw)

            w = w / np.sum(w)
            ess = 1.0 / np.sum(w * w)

            if ess < resample_threshold * nparticles:
                idx = _systematic_resample(w, rng)
                x1 = x1_next[idx]
                x2 = x2_next[idx]
            else:
                x1, x2 = x1_next, x2_next

        return float(loglik)

    def impulse_response(self, para, h: int = 20, use_cache: bool = False, *args, **kwargs):
        """
        Pruned nonlinear impulse responses to 1-s.d. shocks (Dynare-style).
        """
        pol = self._policy(para, use_cache=use_cache)
        _, ZZ, _, QQ = self._measurement_matrices(para, use_cache=use_cache)

        nshocks = QQ.shape[0]
        nendo = pol.gx.shape[0]

        shock_names = self.shock_names or [f"shock_{i}" for i in range(nshocks)]
        endo_names = pol.control_names

        cholQQ = np.linalg.cholesky(QQ)
        irfs = {}

        for i in range(nshocks):
            eps_path = np.zeros((h + 1, nshocks))
            eps_path[0] = cholQQ[:, i]
            y_path = self.perfect_foresight(para, eps_path, include_observables=False, use_cache=use_cache)["states"]
            irfs[shock_names[i]] = y_path[endo_names]

        return irfs

    def perfect_foresight(
        self,
        para,
        eps_path: np.ndarray,
        s0=None,
        include_observables: bool = True,
        use_cache: bool = False,
        *args,
        **kwargs,
    ):
        """
        Deterministic simulation given a shock path using the pruned decision rule.

        Returns a dict with keys 'states' (endo vars) and optionally 'observables'.
        """
        pol = self._policy(para, use_cache=use_cache)
        DD, ZZ, HH, QQ = self._measurement_matrices(para, use_cache=use_cache)

        eps_path = np.asarray(eps_path, dtype=float)
        tmax = eps_path.shape[0]
        nstate = pol.hx.shape[0]

        x1 = np.zeros(nstate) if s0 is None else np.asarray(s0, dtype=float).reshape(-1)
        x2 = np.zeros(nstate)

        endo = np.zeros((tmax, pol.gx.shape[0]))
        obs = np.zeros((tmax, DD.size))

        for t in range(tmax):
            eps = eps_path[t]
            y1 = pol.gx @ x1 + pol.gu @ eps
            y2 = pol.gss + pol.gx @ x2
            if self.pruning:
                y2 = (
                    y2
                    + 0.5 * np.einsum("kij,i,j->k", pol.gxx, x1, x1)
                    + np.einsum("kij,i,j->k", pol.gxu, x1, eps)
                    + 0.5 * np.einsum("kij,i,j->k", pol.guu, eps, eps)
                )
            y = y1 + y2
            endo[t] = y
            obs[t] = DD + ZZ @ y

            x1_next = pol.hx @ x1 + pol.hu @ eps
            x2_next = pol.hss + pol.hx @ x2
            if self.pruning:
                x2_next = (
                    x2_next
                    + 0.5 * np.einsum("kij,i,j->k", pol.hxx, x1, x1)
                    + np.einsum("kij,i,j->k", pol.hxu, x1, eps)
                    + 0.5 * np.einsum("kij,i,j->k", pol.huu, eps, eps)
                )
            x1, x2 = x1_next, x2_next

        endo_df = p.DataFrame(endo, columns=pol.control_names)
        out = {"states": endo_df}
        if include_observables:
            obs_df = p.DataFrame(obs, columns=self.obs_names)
            out["observables"] = obs_df
        return out

    def simulate(self, para, nsim: int = 200, seed: int | None = None, *args, **kwargs):
        """
        Simulate observables using the pruned decision rule with Gaussian shocks and measurement errors.
        """
        pol = self._policy(para, use_cache=False)
        DD, ZZ, HH, QQ = self._measurement_matrices(para, use_cache=False)

        rng = np.random.default_rng(seed)
        nobs = DD.size
        nshocks = QQ.shape[0]

        cholQQ = np.linalg.cholesky(QQ)
        cholHH = np.linalg.cholesky(_ensure_cov_matrix(HH))

        burn = nsim
        tmax = nsim + burn
        eps = rng.standard_normal(size=(tmax, nshocks)) @ cholQQ.T
        meas = rng.standard_normal(size=(tmax, nobs)) @ cholHH.T

        sim = self.perfect_foresight(para, eps, include_observables=True)["observables"].values
        sim = sim + meas
        return sim[burn:, :]

