from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as p


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
    quad = np.einsum("ni,ij,nj->n", resid, inv_cov, resid)
    n_obs = resid.shape[1]
    return -0.5 * (quad + logdet_cov + n_obs * np.log(2.0 * np.pi))


def _logsumexp_1d(logx: np.ndarray) -> float:
    logx = np.asarray(logx, dtype=float).reshape(-1)
    m = float(np.max(logx))
    if not np.isfinite(m):
        return -np.inf
    return m + float(np.log(np.sum(np.exp(logx - m))))


def _ensure_cov_matrix(hh, jitter: float = 1e-12) -> np.ndarray:
    hh = np.asarray(hh, dtype=float)
    if hh.ndim == 0:
        hh = np.array([[float(hh)]])
    elif hh.ndim == 1:
        hh = np.diag(hh)

    if hh.shape[0] != hh.shape[1]:
        raise ValueError(f"Covariance must be square, got {hh.shape}.")

    if np.allclose(hh, 0.0):
        hh = hh + jitter * np.eye(hh.shape[0])
    return hh


def _cov_factor_psd(cov: np.ndarray, *, sym_jitter: float = 0.0) -> np.ndarray:
    """
    Return a factor A such that A A' ≈ cov for PSD cov.

    This is intended for simulation/propagation. It does NOT add noise when cov is all zeros.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    elif cov.ndim == 1:
        cov = np.diag(cov)

    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance must be square, got {cov.shape}.")

    if np.allclose(cov, 0.0):
        return np.zeros_like(cov)

    # Ensure symmetry.
    cov = 0.5 * (cov + cov.T)
    if sym_jitter:
        cov = cov + float(sym_jitter) * np.eye(cov.shape[0])

    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(cov)
        w = np.clip(w, 0.0, None)
        return v @ np.diag(np.sqrt(w))


def _params_cache_key(params: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(params, dtype=float))
    # 64-bit keyed digest to keep keys compact.
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


@dataclass(frozen=True)
class LinearMarginalCostSchedule:
    """
    Linear marginal planning costs: Δτ_{k+1} = a + b (k+1), with a>0, b>=0.
    """

    a: float
    b: float = 0.0

    def __post_init__(self):
        if not (self.a > 0):
            raise ValueError(f"Cost schedule requires a>0, got a={self.a}")
        if not (self.b >= 0):
            raise ValueError(f"Cost schedule requires b>=0, got b={self.b}")

    def delta_tau(self, k_plus_1: int) -> float:
        k_plus_1 = int(k_plus_1)
        if k_plus_1 < 1:
            raise ValueError(f"k_plus_1 must be >= 1, got {k_plus_1}")
        return float(self.a + self.b * k_plus_1)

    def validate_positive(self, k_max: int) -> None:
        # Validate positivity for k in [0..k_max] (i.e. k_plus_1 in [1..k_max+1]).
        k_max = int(k_max)
        if k_max < 0:
            raise ValueError(f"k_max must be >= 0, got {k_max}")
        for k_plus_1 in range(1, k_max + 2):
            val = self.delta_tau(k_plus_1)
            if not (val > 0):
                raise ValueError(
                    f"Cost schedule must have Δτ(k+1)>0. Got Δτ({k_plus_1})={val}."
                )


def pack_regime(components: Sequence[str], k_by_component: Mapping[str, int]) -> Tuple[int, ...]:
    return tuple(int(k_by_component[c]) for c in components)


def unpack_regime(components: Sequence[str], regime: Sequence[int]) -> Dict[str, int]:
    if len(regime) != len(components):
        raise ValueError(f"Regime length {len(regime)} does not match components {len(components)}.")
    return {c: int(regime[i]) for i, c in enumerate(components)}


def choose_k_star(
    *,
    params: np.ndarray,
    info_t: Any,
    component: str,
    k_max: int,
    mb: Callable[[np.ndarray, Any, str, int, Mapping[str, int]], float],
    cost: LinearMarginalCostSchedule,
    chosen: Mapping[str, int] | None = None,
) -> int:
    """
    Stopping rule:
      k* = min{k>=0 : MB(k+1) < Δτ_{k+1}}, capped at k_max.

    Implementation: check k=0..k_max-1, else return k_max.
    """
    k_max = int(k_max)
    if k_max < 0:
        raise ValueError(f"k_max must be >= 0, got {k_max}")
    chosen = {} if chosen is None else dict(chosen)

    for k in range(0, k_max):
        k_plus_1 = k + 1
        mb_val = float(mb(params, info_t, component, k_plus_1, chosen))
        if mb_val < 0:
            raise ValueError(f"MB must be nonnegative, got {mb_val} for {component} at k+1={k_plus_1}")
        if mb_val < cost.delta_tau(k_plus_1):
            return k
    return k_max


def mb_quadratic_policy_diff(
    *,
    params: np.ndarray,
    info_t: Any,
    component: str,
    k_plus_1: int,
    chosen: Mapping[str, int],
    policy_object: Callable[[np.ndarray, Any, str, int, Mapping[str, int]], Any],
    lam: float,
) -> float:
    """
    MB(k+1) = (Λ/2) ||π(k+1) - π(k)||^2, where π(k) = policy_object(..., k).
    """
    if k_plus_1 < 1:
        raise ValueError(f"k_plus_1 must be >= 1, got {k_plus_1}")
    if not (lam > 0):
        raise ValueError(f"Λ must be > 0, got {lam}")

    pi_next = policy_object(params, info_t, component, int(k_plus_1), chosen)
    pi_curr = policy_object(params, info_t, component, int(k_plus_1) - 1, chosen)

    d = np.asarray(pi_next, dtype=float) - np.asarray(pi_curr, dtype=float)
    return float(0.5 * lam * np.sum(d * d))


class RegimeSolutionCache:
    def __init__(
        self,
        *,
        solve_given_regime: Callable[[np.ndarray, Tuple[int, ...]], Tuple[np.ndarray, ...]],
    ):
        self._solve_given_regime = solve_given_regime
        self._cache: Dict[Tuple[str, Tuple[int, ...]], Tuple[np.ndarray, ...]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def get_mats(
        self,
        params: np.ndarray,
        regime: Tuple[int, ...],
        *,
        params_key: str | None = None,
    ) -> Tuple[np.ndarray, ...]:
        if params_key is None:
            params_key = _params_cache_key(params)
        key = (params_key, tuple(int(x) for x in regime))
        cached = self._cache.get(key)
        if cached is None:
            cached = self._solve_given_regime(params, key[1])
            self._cache[key] = cached
        return cached


class EndogenousHorizonSwitchingModel:
    """
    Endogenous regime switching where the regime is a vector of per-component horizons.

    Conditional on regime s_t, the system is linear-Gaussian:
      x_{t+1} = T(s_t) x_t + R(s_t) eps_{t+1},  eps ~ N(0,Q)
      y_t     = Z(s_t) x_t + D(s_t) + eta_t,    eta ~ N(0,H)
    """

    def __init__(
        self,
        *,
        components: Sequence[str],
        k_max: Mapping[str, int] | int,
        cost_params: Mapping[str, Tuple[float, float]] | Tuple[float, float],
        lam: Mapping[str, float] | float,
        cost_func: Callable[[np.ndarray, str], Any] | None = None,
        lam_func: Callable[[np.ndarray, str], Any] | None = None,
        solve_given_regime: Callable[[np.ndarray, Tuple[int, ...]], Tuple[np.ndarray, ...]],
        policy_object: Callable[[np.ndarray, Any, str, int, Mapping[str, int]], Any],
        info_func: Optional[Callable[[np.ndarray, int, Mapping[str, int]], Any]] = None,
        mb_func: Optional[Callable[[np.ndarray, Any, str, int, Mapping[str, int]], float]] = None,
        selection_order: Optional[Sequence[str]] = None,
    ):
        self.components = list(components)
        if not self.components:
            raise ValueError("components must be non-empty")
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"components contains duplicates: {self.components}")

        self.selection_order = list(selection_order) if selection_order is not None else list(self.components)
        if set(self.selection_order) != set(self.components):
            raise ValueError("selection_order must be a permutation of components")

        if isinstance(k_max, int):
            self.k_max = {c: int(k_max) for c in self.components}
        else:
            self.k_max = {c: int(k_max[c]) for c in self.components}
        for c, km in self.k_max.items():
            if km < 0:
                raise ValueError(f"k_max[{c!r}] must be >= 0, got {km}")

        if isinstance(cost_params, tuple):
            default_cost = LinearMarginalCostSchedule(*cost_params)
            self.cost = {c: default_cost for c in self.components}
        else:
            self.cost = {}
            for c in self.components:
                if c in cost_params:
                    self.cost[c] = LinearMarginalCostSchedule(*cost_params[c])
                elif "default" in cost_params:
                    self.cost[c] = LinearMarginalCostSchedule(*cost_params["default"])  # type: ignore[index]
                else:
                    raise ValueError(f"Missing cost_params for component {c!r} (and no 'default').")

        for c in self.components:
            self.cost[c].validate_positive(self.k_max[c])

        if isinstance(lam, (int, float)):
            self.lam = {c: float(lam) for c in self.components}
        else:
            self.lam = {c: float(lam[c]) for c in self.components}
        for c, v in self.lam.items():
            if not (v > 0):
                raise ValueError(f"Λ for component {c!r} must be > 0, got {v}")

        self._cost_func = cost_func
        self._lam_func = lam_func
        self._cost_cache: Dict[Tuple[str, str], LinearMarginalCostSchedule] = {}
        self._lam_cache: Dict[Tuple[str, str], float] = {}
        self._mb_is_default = mb_func is None

        self.policy_object = policy_object
        self.info_func = info_func or (lambda x_t, t, chosen: {"x": x_t, "t": t, "chosen": dict(chosen)})
        if mb_func is None:
            self.mb = lambda params, info_t, component, k_plus_1, chosen: mb_quadratic_policy_diff(
                params=params,
                info_t=info_t,
                component=component,
                k_plus_1=k_plus_1,
                chosen=chosen,
                policy_object=self.policy_object,
                lam=self.lam[component],
            )
        else:
            self.mb = mb_func

        self._cache = RegimeSolutionCache(solve_given_regime=solve_given_regime)

    def _cost_for_component(
        self,
        params: np.ndarray,
        component: str,
        *,
        params_key: str,
    ) -> LinearMarginalCostSchedule:
        if self._cost_func is None:
            return self.cost[component]
        key = (params_key, component)
        cached = self._cost_cache.get(key)
        if cached is not None:
            return cached

        raw = self._cost_func(np.asarray(params, dtype=float), str(component))
        if isinstance(raw, (int, float, np.number)):
            a, b = float(raw), 0.0
        else:
            a, b = raw  # type: ignore[misc]
            a, b = float(a), float(b)
        sched = LinearMarginalCostSchedule(a, b)
        self._cost_cache[key] = sched
        return sched

    def _lam_for_component(self, params: np.ndarray, component: str, *, params_key: str) -> float:
        if self._lam_func is None:
            return self.lam[component]
        key = (params_key, component)
        cached = self._lam_cache.get(key)
        if cached is not None:
            return cached
        v = float(self._lam_func(np.asarray(params, dtype=float), str(component)))
        if not (v > 0):
            raise ValueError(f"Λ for component {component!r} must be > 0, got {v}")
        self._lam_cache[key] = v
        return v

    def pack_regime(self, k_by_component: Mapping[str, int]) -> Tuple[int, ...]:
        return pack_regime(self.components, k_by_component)

    def unpack_regime(self, regime: Sequence[int]) -> Dict[str, int]:
        return unpack_regime(self.components, regime)

    def warm_cache(self, params: np.ndarray, *, max_regimes: int = 50_000) -> None:
        sizes = [self.k_max[c] + 1 for c in self.components]
        nreg = int(np.prod(sizes))
        if nreg > max_regimes:
            raise ValueError(f"Refusing to warm_cache {nreg} regimes (max_regimes={max_regimes}).")
        pkey = _params_cache_key(params)
        for tup in product(*[range(s) for s in sizes]):
            self._cache.get_mats(params, tuple(int(x) for x in tup), params_key=pkey)

    def get_mats(
        self,
        params: np.ndarray,
        regime: Tuple[int, ...],
        *,
        params_key: str | None = None,
    ) -> Tuple[np.ndarray, ...]:
        return self._cache.get_mats(params, regime, params_key=params_key)

    def choose_regime(
        self,
        params: np.ndarray,
        x_t: np.ndarray,
        *,
        t: int,
        params_key: str | None = None,
    ) -> Tuple[int, ...]:
        if params_key is None:
            params_key = _params_cache_key(params)

        if self._lam_func is not None and self._mb_is_default:
            # When Λ is param-dependent, build a per-call MB closure to avoid hashing
            # params on each MB evaluation.
            def mb_dyn(params_vec, info_t, component, k_plus_1, chosen):
                lam_val = self._lam_for_component(params_vec, component, params_key=params_key)
                return mb_quadratic_policy_diff(
                    params=params_vec,
                    info_t=info_t,
                    component=component,
                    k_plus_1=k_plus_1,
                    chosen=chosen,
                    policy_object=self.policy_object,
                    lam=lam_val,
                )

            mb_use = mb_dyn
        else:
            mb_use = self.mb

        chosen: MutableMapping[str, int] = {}
        for comp in self.selection_order:
            info_t = self.info_func(x_t, int(t), chosen)
            chosen[comp] = choose_k_star(
                params=params,
                info_t=info_t,
                component=comp,
                k_max=self.k_max[comp],
                mb=mb_use,
                cost=self._cost_for_component(params, comp, params_key=params_key),
                chosen=chosen,
            )
        return self.pack_regime(chosen)

    def simulate(
        self,
        params: np.ndarray,
        T: int,
        *,
        seed: int | None = None,
        x0: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        params = np.asarray(params, dtype=float)
        pkey = _params_cache_key(params)

        # Determine dims from an arbitrary regime (all zeros).
        regime0 = tuple(0 for _ in self.components)
        TT, RR, ZZ, DD, QQ, HH = self.get_mats(params, regime0, params_key=pkey)

        nstate = int(TT.shape[0])
        nshock = int(QQ.shape[0])
        nobs = int(ZZ.shape[0])

        x_t = np.zeros((nstate,), dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        if x_t.shape != (nstate,):
            raise ValueError(f"x0 must have shape ({nstate},), got {x_t.shape}")

        x_path = np.zeros((T + 1, nstate), dtype=float)
        y_path = np.zeros((T, nobs), dtype=float)
        s_path = np.zeros((T, len(self.components)), dtype=int)

        x_path[0] = x_t

        for t in range(T):
            s_t = self.choose_regime(params, x_t, t=t, params_key=pkey)
            TT, RR, ZZ, DD, QQ, HH = self.get_mats(params, s_t, params_key=pkey)

            mean_y = ZZ @ x_t + np.asarray(DD, dtype=float).reshape(-1)
            cholHH = _cov_factor_psd(HH)
            eta = rng.standard_normal(size=(nobs,)) @ cholHH.T
            y_t = mean_y + eta

            s_path[t, :] = np.asarray(s_t, dtype=int)
            y_path[t, :] = y_t

            cholQQ = _cov_factor_psd(QQ)
            eps = rng.standard_normal(size=(nshock,)) @ cholQQ.T
            x_next = TT @ x_t + RR @ eps

            x_path[t + 1] = x_next
            x_t = x_next

        return {"x_path": x_path, "y_path": y_path, "s_path": s_path}

    def girf(
        self,
        params: np.ndarray,
        *,
        shock: str | int | None = None,
        h: int = 20,
        reps: int = 200,
        shock_size: float = 1.0,
        seed: int | None = None,
        x0: Optional[np.ndarray] = None,
        obs_subset: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generalized impulse response (GIRF) for endogenous regime switching.

        Implementation: Monte Carlo with common random numbers baseline vs shocked,
        applying the shock as an innovation at t=0 (so effects appear starting at t=1).

        Returns a dict with keys:
          - girf: DataFrame (h x nobs), mean(y_shocked - y_base)
          - k_base_mean, k_shocked_mean: DataFrame (h x ncomp), mean chosen horizons
        """
        params = np.asarray(params, dtype=float)
        pkey = _params_cache_key(params)
        rng = np.random.default_rng(seed)

        regime0 = tuple(0 for _ in self.components)
        TT0, RR0, ZZ0, DD0, QQ0, HH0 = self.get_mats(params, regime0, params_key=pkey)
        QQ0 = np.asarray(QQ0, dtype=float)

        nstate = int(TT0.shape[0])
        nshock = int(QQ0.shape[0])
        nobs = int(ZZ0.shape[0])

        x0_vec = np.zeros((nstate,), dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        if x0_vec.shape != (nstate,):
            raise ValueError(f"x0 must have shape ({nstate},), got {x0_vec.shape}")

        shock_names = getattr(self, "shock_names", None)
        if shock_names is None:
            shock_names = [f"shock{i}" for i in range(nshock)]

        if shock is None:
            shock_idx = 0
            shock_name = shock_names[0]
        elif isinstance(shock, int):
            shock_idx = int(shock)
            if not (0 <= shock_idx < nshock):
                raise ValueError(f"shock index must be in [0,{nshock}), got {shock_idx}")
            shock_name = shock_names[shock_idx]
        else:
            shock_name = str(shock)
            if shock_name not in list(shock_names):
                raise ValueError(f"Unknown shock {shock_name!r}. Expected one of {list(shock_names)}.")
            shock_idx = int(list(shock_names).index(shock_name))

        var0 = float(QQ0[shock_idx, shock_idx])
        shock_delta = np.zeros((nshock,), dtype=float)
        shock_delta[shock_idx] = float(shock_size) * (np.sqrt(var0) if var0 > 0 else 1.0)

        cholQQ = _cov_factor_psd(QQ0)

        y_diff_sum = np.zeros((h, nobs), dtype=float)
        k_base_sum = np.zeros((h, len(self.components)), dtype=float)
        k_shock_sum = np.zeros((h, len(self.components)), dtype=float)

        def _simulate_eps(*, eps_path: np.ndarray, shock_delta_t0: np.ndarray | None):
            x_t = x0_vec.copy()
            y_path = np.zeros((h, nobs), dtype=float)
            s_path = np.zeros((h, len(self.components)), dtype=int)
            for t in range(h):
                s_t = self.choose_regime(params, x_t, t=t, params_key=pkey)
                TT, RR, ZZ, DD, QQ, HH = self.get_mats(params, s_t, params_key=pkey)
                y_path[t, :] = ZZ @ x_t + np.asarray(DD, dtype=float).reshape(-1)
                s_path[t, :] = np.asarray(s_t, dtype=int)
                eps_t = eps_path[t, :]
                if t == 0 and shock_delta_t0 is not None:
                    eps_t = eps_t + shock_delta_t0
                x_t = TT @ x_t + RR @ eps_t
            return y_path, s_path

        for _ in range(int(reps)):
            z = rng.standard_normal(size=(h, nshock))
            eps = z @ cholQQ.T

            y_base, s_base = _simulate_eps(eps_path=eps, shock_delta_t0=None)
            y_shock, s_shock = _simulate_eps(eps_path=eps, shock_delta_t0=shock_delta)

            y_diff_sum += y_shock - y_base
            k_base_sum += s_base
            k_shock_sum += s_shock

        obs_names = getattr(self, "obs_names", None)
        if obs_names is None:
            obs_names = [f"obs{i}" for i in range(nobs)]

        girf = p.DataFrame(y_diff_sum / float(reps), columns=list(obs_names))
        k_base_mean = p.DataFrame(k_base_sum / float(reps), columns=list(self.components))
        k_shocked_mean = p.DataFrame(k_shock_sum / float(reps), columns=list(self.components))

        if obs_subset is not None:
            want = [str(v) for v in obs_subset]
            missing = sorted(set(want) - set(girf.columns))
            if missing:
                raise ValueError(f"obs_subset has unknown names: {missing}. Available: {list(girf.columns)}.")
            girf = girf[want]

        return {
            "shock": shock_name,
            "girf": girf,
            "k_base_mean": k_base_mean,
            "k_shocked_mean": k_shocked_mean,
        }

    def pf_loglik(
        self,
        params: np.ndarray,
        y_data,
        *,
        nparticles: int = 2000,
        seed: int | None = None,
        resample_threshold: float = 0.5,
        x0: Optional[np.ndarray] = None,
        x0_cov: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        params = np.asarray(params, dtype=float)
        pkey = _params_cache_key(params)

        y = _as_2d_array(y_data)
        Tobs, nobs = y.shape

        regime0 = tuple(0 for _ in self.components)
        TT0, RR0, ZZ0, DD0, QQ0, HH0 = self.get_mats(params, regime0, params_key=pkey)
        nstate = int(TT0.shape[0])

        x_mean0 = np.zeros((nstate,), dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        if x_mean0.shape != (nstate,):
            raise ValueError(f"x0 must have shape ({nstate},), got {x_mean0.shape}")

        if x0_cov is None:
            x = np.tile(x_mean0, (nparticles, 1))
        else:
            x0_cov = np.asarray(x0_cov, dtype=float)
            if x0_cov.shape != (nstate, nstate):
                raise ValueError(f"x0_cov must have shape ({nstate},{nstate}), got {x0_cov.shape}")
            chol0 = np.linalg.cholesky(_ensure_cov_matrix(x0_cov))
            x = x_mean0[None, :] + rng.standard_normal(size=(nparticles, nstate)) @ chol0.T

        logw = np.full(nparticles, -np.log(nparticles), dtype=float)
        loglik = 0.0

        # Cache inverses for (regime, mask) pairs.
        cov_cache: Dict[Tuple[Tuple[int, ...], bytes], Tuple[np.ndarray, float]] = {}

        k_mean = np.zeros((Tobs, len(self.components)), dtype=float)
        x_filt_mean = np.zeros((Tobs, nstate), dtype=float)

        for t in range(Tobs):
            # Choose regime per particle.
            regimes = [self.choose_regime(params, x[i, :], t=t, params_key=pkey) for i in range(nparticles)]
            regimes_arr = np.asarray(regimes, dtype=int)

            # Measurement update.
            y_t = y[t]
            mask = np.isfinite(y_t)
            if np.any(mask):
                # Group by regime to avoid redundant mat fetches.
                idx_by_reg_list: Dict[Tuple[int, ...], list[int]] = {}
                for i, r in enumerate(regimes):
                    idx_by_reg_list.setdefault(r, []).append(i)
                idx_by_reg = {r: np.asarray(ix, dtype=int) for r, ix in idx_by_reg_list.items()}

                logp = np.empty((nparticles,), dtype=float)
                for r, ix in idx_by_reg.items():
                    TT, RR, ZZ, DD, QQ, HH = self.get_mats(params, r, params_key=pkey)
                    mean = x[ix, :] @ ZZ.T + np.asarray(DD, dtype=float).reshape(1, -1)
                    resid = y_t[mask][None, :] - mean[:, mask]

                    mask_key = mask.tobytes()
                    cache_key = (r, mask_key)
                    cached = cov_cache.get(cache_key)
                    if cached is None:
                        HH_use = _ensure_cov_matrix(HH)
                        HH_t = HH_use[np.ix_(mask, mask)]
                        sign_t, logdet_t = np.linalg.slogdet(HH_t)
                        if sign_t <= 0:
                            raise ValueError("Measurement covariance must be positive definite.")
                        cached = (np.linalg.inv(HH_t), float(logdet_t))
                        cov_cache[cache_key] = cached
                    invHH_t, logdet_t = cached
                    logp[ix] = _logpdf_mvnormal(resid, invHH_t, logdet_t)

                logw = logw + logp
                inc = _logsumexp_1d(logw)
                if not np.isfinite(inc):
                    return -np.inf, {"k_mean": k_mean, "x_mean": x_filt_mean}
                loglik += inc
                logw = logw - inc
            else:
                # Missing observation: no weight update.
                pass

            w = np.exp(logw)
            k_mean[t, :] = w @ regimes_arr
            x_filt_mean[t, :] = w @ x

            ess = 1.0 / float(np.sum(w * w))
            if ess < resample_threshold * nparticles:
                idx = _systematic_resample(w, rng)
                x = x[idx, :]
                regimes_arr = regimes_arr[idx, :]
                logw.fill(-np.log(nparticles))
            else:
                # Keep regimes_arr as-is; logw already normalized.
                pass

            # Propagate to next state using selected regimes.
            if t < Tobs - 1:
                x_next = np.empty_like(x)

                # regroup based on regimes after resampling
                idx_by_reg_list = {}
                for i in range(nparticles):
                    r = tuple(int(v) for v in regimes_arr[i, :])
                    idx_by_reg_list.setdefault(r, []).append(i)
                idx_by_reg = {r: np.asarray(ix, dtype=int) for r, ix in idx_by_reg_list.items()}

                for r, ix in idx_by_reg.items():
                    TT, RR, ZZ, DD, QQ, HH = self.get_mats(params, r, params_key=pkey)
                    QQ_use = np.asarray(QQ, dtype=float)
                    cholQQ = _cov_factor_psd(QQ_use)
                    nshock = int(cholQQ.shape[0])
                    eps = rng.standard_normal(size=(ix.size, nshock)) @ cholQQ.T
                    x_next[ix, :] = x[ix, :] @ TT.T + eps @ RR.T

                x = x_next

        stats = {"k_mean": k_mean, "x_mean": x_filt_mean}
        return float(loglik), stats
