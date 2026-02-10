from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np


BAD_LOG_LIKELIHOOD = -1e11
BAD_LOG_PRIOR = -1e11


class PriorLike(Protocol):
    priors: Optional[Sequence[Any]]

    def logpdf(self, para: np.ndarray) -> float: ...

    def rvs(self, size: Optional[int] = None, random_state: Any = None) -> np.ndarray: ...


class ModelLike(Protocol):
    prior: Optional[PriorLike]
    parameter_names: Optional[Sequence[str]]

    def log_lik(self, para: np.ndarray, use_cache: bool = False, *args: Any, **kwargs: Any) -> float: ...


def _phi_schedule(nphi: int, bend: float, *, phi_max: float = 1.0) -> np.ndarray:
    if nphi < 2:
        raise ValueError("nphi must be >= 2 so the schedule reaches phi=1.")
    phi = phi_max * (np.arange(nphi, dtype=float) / (nphi - 1.0))
    phi = np.power(phi, float(bend))
    phi[-1] = phi_max
    return phi


def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float).reshape(-1)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        return 0.0
    w = w / s
    return float(1.0 / np.sum(w * w))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w / float(np.sum(w))
    n = w.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative_sum = np.cumsum(w)
    idx = np.searchsorted(cumulative_sum, positions, side="right")
    return idx


def _log_prior_many(model: ModelLike, particles: np.ndarray) -> np.ndarray:
    prior = getattr(model, "prior", None)
    if prior is None or getattr(prior, "priors", None) is None:
        return np.zeros((particles.shape[0],), dtype=float)

    dists = list(prior.priors)  # type: ignore[arg-type]
    if particles.shape[1] != len(dists):
        raise ValueError(
            f"Prior dimension mismatch: particles have npara={particles.shape[1]} but prior has {len(dists)} entries."
        )

    lp = np.zeros((particles.shape[0],), dtype=float)
    for j, dist in enumerate(dists):
        lp = lp + np.asarray(dist.logpdf(particles[:, j]), dtype=float)
    lp = np.where(np.isfinite(lp), lp, BAD_LOG_PRIOR)
    lp = np.where(lp == -np.inf, BAD_LOG_PRIOR, lp)
    return lp


def _chunk_slices(n: int, *, n_workers: int, chunks_per_worker: int = 4) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    n_workers = int(max(1, n_workers))
    chunks_per_worker = int(max(1, chunks_per_worker))
    n_tasks = min(n, n_workers * chunks_per_worker)
    chunk = int((n + n_tasks - 1) // n_tasks)
    return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]


class _LogLikBatchEvaluator:
    def __init__(
        self,
        model: ModelLike,
        *,
        log_lik_kwargs: Mapping[str, Any],
        n_workers: int,
        parallel: Literal["none", "thread"],
    ):
        self._model = model
        self._log_lik_kwargs = dict(log_lik_kwargs)
        self._parallel = str(parallel)
        self._n_workers = int(max(1, n_workers))
        self._executor = None

        if self._parallel not in {"none", "thread"}:
            raise ValueError(f"Unsupported parallel mode: {parallel!r}.")

        if self._parallel == "thread" and self._n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            self._executor = ThreadPoolExecutor(max_workers=self._n_workers)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _eval_one(self, p: np.ndarray) -> float:
        try:
            ll = float(self._model.log_lik(p, use_cache=False, **self._log_lik_kwargs))
        except Exception:
            return BAD_LOG_LIKELIHOOD
        if not np.isfinite(ll):
            return BAD_LOG_LIKELIHOOD
        return ll

    def eval_many(self, particles: np.ndarray) -> np.ndarray:
        particles = np.asarray(particles, dtype=float)
        if particles.ndim != 2:
            raise ValueError("particles must be a 2D array (npart, npara).")
        n = int(particles.shape[0])

        if self._executor is None:
            out = np.empty((n,), dtype=float)
            for i in range(n):
                out[i] = self._eval_one(particles[i, :])
            return out

        # Chunk work to amortize Python scheduling overhead.
        slices = _chunk_slices(n, n_workers=self._n_workers, chunks_per_worker=1)

        def _eval_slice(bounds: tuple[int, int]) -> tuple[int, np.ndarray]:
            lo, hi = bounds
            buf = np.empty((hi - lo,), dtype=float)
            for j in range(lo, hi):
                buf[j - lo] = self._eval_one(particles[j, :])
            return lo, buf

        out = np.empty((n,), dtype=float)
        for lo, buf in self._executor.map(_eval_slice, slices):
            out[lo : lo + buf.size] = buf
        return out


def _weighted_mean_and_cov(particles: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(particles, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w / float(np.sum(w))
    mean = np.sum(x * w[:, None], axis=0)
    xc = x - mean
    cov = (xc.T * w) @ xc
    return mean, cov


def _blocks(rng: np.random.Generator, npara: int, nblocks: int) -> list[np.ndarray]:
    if nblocks < 1:
        raise ValueError("nblocks must be >= 1.")
    idx = np.arange(npara, dtype=int)
    if nblocks == 1:
        return [idx]
    rng.shuffle(idx)
    return [np.asarray(b, dtype=int) for b in np.array_split(idx, nblocks)]


def _block_cholesky_factors(
    cov: np.ndarray,
    blocks: Sequence[np.ndarray],
    *,
    conditional_covariance: bool,
    jitter: float = 1e-10,
) -> list[np.ndarray]:
    cov = np.asarray(cov, dtype=float)
    npara = cov.shape[0]
    if cov.shape != (npara, npara):
        raise ValueError(f"cov must be square, got {cov.shape}.")

    factors: list[np.ndarray] = []
    for b in blocks:
        b = np.asarray(b, dtype=int)
        bb = cov[np.ix_(b, b)].copy()

        if conditional_covariance and len(blocks) > 1:
            rest = np.setdiff1d(np.arange(npara, dtype=int), b, assume_unique=False)
            if rest.size > 0:
                br = cov[np.ix_(b, rest)]
                rr = cov[np.ix_(rest, rest)]
                try:
                    rr_inv = np.linalg.inv(rr)
                except np.linalg.LinAlgError:
                    rr_inv = np.linalg.pinv(rr)
                bb = bb - br @ rr_inv @ br.T

        # Numerical safety
        bb = 0.5 * (bb + bb.T)
        bb = bb + jitter * np.eye(bb.shape[0])
        try:
            L = np.linalg.cholesky(bb)
        except np.linalg.LinAlgError:
            # Escalate jitter until PSD.
            scale = 1.0
            while True:
                try:
                    L = np.linalg.cholesky(bb + (scale * jitter) * np.eye(bb.shape[0]))
                    break
                except np.linalg.LinAlgError:
                    scale *= 10.0
                    if scale > 1e8:
                        raise
        factors.append(L)
    return factors


def _choose_phi_endogenous(
    *,
    phi_old: float,
    phi_max: float,
    log_lik: np.ndarray,
    weights: np.ndarray,
    ess_target: float,
    tol: float,
) -> float:
    phi_old = float(phi_old)
    phi_max = float(phi_max)
    if phi_old >= phi_max:
        return phi_max

    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w / float(np.sum(w))
    ll = np.asarray(log_lik, dtype=float).reshape(-1)

    def ess_at(phi: float) -> float:
        delta = (float(phi) - phi_old) * ll
        m = float(np.max(delta))
        if not np.isfinite(m):
            return 0.0
        w_unnorm = w * np.exp(delta - m)
        z = float(np.sum(w_unnorm))
        if z <= 0.0 or not np.isfinite(z):
            return 0.0
        w_new = w_unnorm / z
        return float(1.0 / np.sum(w_new * w_new))

    ess_max = ess_at(phi_max)
    if ess_max >= ess_target:
        return phi_max

    lo, hi = phi_old, phi_max
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        ess_mid = ess_at(mid)
        if abs(ess_mid - ess_target) <= tol * ess_target:
            return mid
        if ess_mid < ess_target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12:
            return mid
    return 0.5 * (lo + hi)


@dataclass
class SMCOptions:
    """
    Tempered SMC options (Fortress-like).

    Notes
    -----
    - `phi_schedule` is `((i-1)/(nphi-1))**bend` with endpoints {0, 1}.
    - Only `mutation_type="RWMH"` is implemented currently.
    """

    npart: int = 4800
    nphi: int = 500
    bend: float = 2.0
    seed: int = 1848

    mutation_type: str = "RWMH"
    nintmh: int = 1
    nblocks: int = 1
    conditional_covariance: bool = False

    initial_scale: float = 0.4
    target_accept: float = 0.25

    resample_threshold: float = 0.5
    resample_every_stage: bool = False

    endog_tempering: bool = False
    resample_tol: float = 0.5
    bisection_thresh: float = 1e-3

    npriorextra: int = 1000
    verbose: bool = True

    def phi_schedule(self) -> np.ndarray:
        return _phi_schedule(self.nphi, self.bend, phi_max=1.0)


@dataclass
class SMCStageStats:
    phi: float
    ess: float
    logz: float
    accepted: float
    scale: float
    resampled: bool


@dataclass
class SMCResult:
    particles: np.ndarray  # (npart, npara)
    weights: np.ndarray  # (npart,)
    log_lik: np.ndarray  # (npart,)
    log_prior: np.ndarray  # (npart,)
    phi_schedule: np.ndarray  # (nphi,)
    logz_estimates: np.ndarray  # (nphi,)
    ess_estimates: np.ndarray  # (nphi,)
    stage_stats: list[SMCStageStats]
    parameter_names: Optional[list[str]] = None

    def posterior_frame(self):
        import pandas as p

        names = self.parameter_names or [f"var.{i+1:03d}" for i in range(self.particles.shape[1])]
        frame = p.DataFrame(self.particles, columns=list(names))
        frame["weights"] = self.weights
        frame["loglh"] = self.log_lik
        frame["prior"] = self.log_prior
        return frame


def smc_estimate(
    model: ModelLike,
    *,
    options: Optional[SMCOptions] = None,
    log_lik_kwargs: Optional[Mapping[str, Any]] = None,
    n_workers: int = 1,
    parallel: Literal["none", "thread"] = "thread",
) -> SMCResult:
    """
    Run a tempered SMC sampler for a model with `log_lik` + `prior`.

    Parameters
    ----------
    model
        Compiled model (e.g., `LinearDSGEModel`, `PerturbationDSGEModel`).
        Must expose `log_lik(para, ...)` and `prior` with `.rvs` + `.logpdf`.
    options
        SMC tuning parameters.
    log_lik_kwargs
        Extra keyword arguments forwarded to `model.log_lik` (e.g. particle-filter settings
        for order-2 models).
    n_workers, parallel
        Parallel evaluation strategy for batch likelihood calls. Thread-parallelism avoids
        pickling the model (useful for SymPy-lambdified callables).
    """
    if options is None:
        options = SMCOptions()
    if log_lik_kwargs is None:
        log_lik_kwargs = {}

    if options.mutation_type != "RWMH":
        raise NotImplementedError("Only mutation_type='RWMH' is implemented for Python SMC (for now).")

    npart = int(options.npart)
    if npart <= 0:
        raise ValueError("npart must be positive.")

    nintmh = int(options.nintmh)
    if nintmh <= 0:
        raise ValueError("nintmh must be positive.")

    nblocks = int(options.nblocks)
    if nblocks <= 0:
        raise ValueError("nblocks must be positive.")

    rng = np.random.default_rng(int(options.seed))

    prior = getattr(model, "prior", None)
    if prior is None or getattr(prior, "priors", None) is None:
        raise ValueError("Model has no prior attached; SMC requires `model.prior`.")

    with _LogLikBatchEvaluator(
        model,
        log_lik_kwargs=log_lik_kwargs,
        n_workers=int(n_workers),
        parallel=parallel,
    ) as ll_eval:
        # -----------------------------------------------------------------
        # Initialization: draw from prior + evaluate likelihood once.
        # -----------------------------------------------------------------
        draws_needed = npart + int(options.npriorextra)
        particles_all = np.asarray(prior.rvs(size=draws_needed, random_state=rng), dtype=float)
        if particles_all.ndim != 2:
            raise ValueError("prior.rvs(size=...) must return a 2D array (n, npara).")
        particles_all = particles_all.reshape(draws_needed, -1)
        npara = particles_all.shape[1]

        ll_all = ll_eval.eval_many(particles_all)

        good = np.isfinite(ll_all) & (ll_all > BAD_LOG_LIKELIHOOD)
        if int(np.sum(good)) < npart:
            raise RuntimeError(
                "Prior is too far from likelihood: not enough finite likelihood draws.\n"
                f"- needed npart={npart}, got {int(np.sum(good))}\n"
                f"- consider increasing npriorextra (currently {options.npriorextra}) or adjusting the prior"
            )

        good_idx = np.flatnonzero(good)[:npart]
        particles = particles_all[good_idx, :].copy()
        log_lik = ll_all[good_idx].copy()
        log_prior = _log_prior_many(model, particles)

        weights = np.full((npart,), 1.0 / npart, dtype=float)

        phi_sched = options.phi_schedule()
        logz = np.zeros_like(phi_sched, dtype=float)
        ess = np.zeros_like(phi_sched, dtype=float)
        stage_stats: list[SMCStageStats] = []

        ess[0] = _effective_sample_size(weights)

        scale = float(options.initial_scale)
        target_accept = float(options.target_accept)

        # ---------------------------------------------------------------
        # Main loop over tempering stages.
        # ---------------------------------------------------------------
        for i in range(1, phi_sched.size):
            phi_old = float(phi_sched[i - 1])
            phi = float(phi_sched[i])

            if options.endog_tempering:
                phi = _choose_phi_endogenous(
                    phi_old=phi_old,
                    phi_max=1.0,
                    log_lik=log_lik,
                    weights=weights,
                    ess_target=float(options.resample_tol) * npart,
                    tol=float(options.bisection_thresh),
                )
                phi_sched[i] = phi

            # --------------------------
            # Correction (reweight)
            # --------------------------
            inc = (phi - phi_old) * log_lik
            m = float(np.max(inc))
            if not np.isfinite(m):
                raise RuntimeError("SMC weight update failed: max incremental log-weight is not finite.")
            w_unnorm = weights * np.exp(inc - m)
            zt = float(np.sum(w_unnorm))
            if zt <= 0.0 or not np.isfinite(zt):
                raise RuntimeError("SMC weight update failed: weight normalization constant is invalid.")
            weights = w_unnorm / zt
            logz[i] = np.log(zt) + m

            ess_i = _effective_sample_size(weights)
            ess[i] = ess_i

            # --------------------------
            # Selection (resample)
            # --------------------------
            resampled = False
            if options.resample_every_stage or (ess_i < float(options.resample_threshold) * npart):
                idx = _systematic_resample(weights, rng)
                particles = particles[idx, :]
                log_lik = log_lik[idx]
                log_prior = log_prior[idx]
                weights = np.full((npart,), 1.0 / npart, dtype=float)
                resampled = True

            # --------------------------
            # Mutation (block RWMH)
            # --------------------------
            blocks = _blocks(rng, npara, nblocks)
            _, cov = _weighted_mean_and_cov(particles, weights)
            chols = _block_cholesky_factors(
                cov,
                blocks,
                conditional_covariance=bool(options.conditional_covariance),
            )

            eps = rng.standard_normal(size=(npart, nintmh, npara))
            uu = rng.random(size=(npart, nintmh, len(blocks)))

            accepted_total = 0.0
            for mstep in range(nintmh):
                for b_idx, b in enumerate(blocks):
                    L = chols[b_idx]
                    z = eps[:, mstep, :][:, b]
                    prop = particles.copy()
                    prop[:, b] = prop[:, b] + scale * (z @ L.T)

                    ll_prop = ll_eval.eval_many(prop)
                    lp_prop = _log_prior_many(model, prop)

                    log_alpha = phi * (ll_prop - log_lik) + (lp_prop - log_prior)
                    log_u = np.log(uu[:, mstep, b_idx])
                    accept = np.isfinite(log_alpha) & (log_u < log_alpha)

                    if np.any(accept):
                        particles[accept, :] = prop[accept, :]
                        log_lik[accept] = ll_prop[accept]
                        log_prior[accept] = lp_prop[accept]

                    accepted_total += float(np.sum(accept)) * float(b.size)

            accept_rate = accepted_total / float(npart * nintmh * npara)

            # Fortress-style scale adaptation.
            scale *= 0.80 + 0.40 * (
                np.exp(16.0 * (accept_rate - target_accept))
                / (1.0 + np.exp(16.0 * (accept_rate - target_accept)))
            )

            stage_stats.append(
                SMCStageStats(
                    phi=phi,
                    ess=ess_i,
                    logz=float(logz[i]),
                    accepted=float(accept_rate),
                    scale=float(scale),
                    resampled=bool(resampled),
                )
            )

    param_names = None
    if getattr(model, "parameter_names", None) is not None:
        param_names = list(model.parameter_names)  # type: ignore[arg-type]

    return SMCResult(
        particles=particles,
        weights=weights,
        log_lik=log_lik,
        log_prior=log_prior,
        phi_schedule=phi_sched,
        logz_estimates=logz,
        ess_estimates=ess,
        stage_stats=stage_stats,
        parameter_names=param_names,
    )


def smc_estimate_from_yaml(
    yaml_path: str,
    *,
    order: int = 1,
    options: Optional[SMCOptions] = None,
    log_lik_kwargs: Optional[Mapping[str, Any]] = None,
    n_workers: int = 1,
    parallel: Literal["none", "thread"] = "thread",
) -> SMCResult:
    """
    Convenience wrapper for Slurm/remote runs: load YAML, compile, run SMC.
    """
    from .parse_yaml import read_yaml

    model = read_yaml(yaml_path)
    compiled = model.compile_model(order=order)
    return smc_estimate(
        compiled,
        options=options,
        log_lik_kwargs=log_lik_kwargs,
        n_workers=n_workers,
        parallel=parallel,
    )


def smc_submit_slurm(
    yaml_path: str,
    *,
    order: int = 1,
    options: Optional[SMCOptions] = None,
    log_lik_kwargs: Optional[Mapping[str, Any]] = None,
    n_workers: int = 1,
    parallel: Literal["none", "thread"] = "thread",
    submitit_folder: str = "_submitit",
    slurm_params: Optional[Mapping[str, Any]] = None,
):
    """
    Submit an SMC estimation to Slurm via `submitit`.

    Returns
    -------
    submitit.Job
        A handle that supports `.result()`.
    """
    try:
        import submitit  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "submitit is required for backend='slurm'. Install with `uv pip install submitit` "
            "(or add it to your environment)."
        ) from exc

    if options is None:
        options = SMCOptions()

    executor = submitit.AutoExecutor(folder=str(submitit_folder))
    if slurm_params:
        executor.update_parameters(**dict(slurm_params))
    job = executor.submit(
        smc_estimate_from_yaml,
        yaml_path,
        order=int(order),
        options=options,
        log_lik_kwargs=dict(log_lik_kwargs or {}),
        n_workers=int(n_workers),
        parallel=str(parallel),
    )
    return job
