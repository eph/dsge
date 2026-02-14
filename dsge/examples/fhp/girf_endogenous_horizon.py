#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from dsge import read_yaml


def _cov_factor_psd(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    elif cov.ndim == 1:
        cov = np.diag(cov)

    if np.allclose(cov, 0.0):
        return np.zeros_like(cov)

    cov = 0.5 * (cov + cov.T)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(cov)
        w = np.clip(w, 0.0, None)
        return v @ np.diag(np.sqrt(w))


def _simulate_given_eps(
    model,
    *,
    params: np.ndarray,
    eps_path: np.ndarray,
    shock_delta_t0: np.ndarray | None = None,
) -> Dict[str, Any]:
    params = np.asarray(params, dtype=float)
    eps_path = np.asarray(eps_path, dtype=float)
    if eps_path.ndim != 2:
        raise ValueError(f"eps_path must be 2D (T, nshock), got {eps_path.shape}")

    T = int(eps_path.shape[0])
    components = list(getattr(model, "components", []))
    regime0: Tuple[int, ...] = tuple(0 for _ in components)
    TT0, RR0, ZZ0, DD0, QQ0, HH0 = model.get_mats(params, regime0)

    nstate = int(TT0.shape[0])
    nobs = int(ZZ0.shape[0])
    nshock = int(QQ0.shape[0])
    if eps_path.shape[1] != nshock:
        raise ValueError(f"eps_path has nshock={eps_path.shape[1]}, expected {nshock}")

    x_t = np.zeros((nstate,), dtype=float)
    y_path = np.zeros((T, nobs), dtype=float)
    s_path = np.zeros((T, len(components)), dtype=int)

    for t in range(T):
        s_t = model.choose_regime(params, x_t, t=t)
        TT, RR, ZZ, DD, QQ, HH = model.get_mats(params, s_t)

        y_path[t, :] = ZZ @ x_t + np.asarray(DD, dtype=float).reshape(-1)
        s_path[t, :] = np.asarray(s_t, dtype=int)

        eps_t = eps_path[t, :]
        if t == 0 and shock_delta_t0 is not None:
            eps_t = eps_t + shock_delta_t0
        x_t = TT @ x_t + RR @ eps_t

    return {"y_path": y_path, "s_path": s_path}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generalized IRF for endogenous-horizon FHP (Monte Carlo).")
    ap.add_argument("--yaml", default="dsge/examples/fhp/partial_equilibrium_endogenous.yaml")
    ap.add_argument("--shock", default=None, help="Innovation name (e.g. e_y). Defaults to first shock.")
    ap.add_argument("--h", type=int, default=20, help="Horizon length")
    ap.add_argument("--reps", type=int, default=200, help="Monte Carlo replications")
    ap.add_argument("--shock_size", type=float, default=1.0, help="Size in std deviations")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")
    ap.add_argument("--vars", default=None, help="Comma-separated observables to print (default: all)")
    args = ap.parse_args()

    model = read_yaml(args.yaml)
    if not hasattr(model, "choose_regime") or not hasattr(model, "get_mats"):
        raise SystemExit(
            "YAML did not compile to a switching model (missing declarations.stopping_rule / declarations.horizon_choice?)."
        )

    params = np.asarray(getattr(model, "p0", None), dtype=float)
    if params.size == 0:
        raise SystemExit("Switching model is missing calibration vector 'p0'.")

    components = list(getattr(model, "components", []))
    shock_names = list(getattr(model, "shock_names", [])) or None
    obs_names = list(getattr(model, "obs_names", [])) or None

    regime0: Tuple[int, ...] = tuple(0 for _ in components)
    TT0, RR0, ZZ0, DD0, QQ0, HH0 = model.get_mats(params, regime0)
    QQ0 = np.asarray(QQ0, dtype=float)

    nshock = int(QQ0.shape[0])
    if shock_names is None:
        shock_names = [f"shock{i}" for i in range(nshock)]
    if len(shock_names) != nshock:
        raise SystemExit(f"shock_names length {len(shock_names)} != nshock {nshock}")

    if obs_names is None:
        obs_names = [f"obs{i}" for i in range(int(ZZ0.shape[0]))]

    shock = args.shock if args.shock is not None else shock_names[0]
    if shock not in shock_names:
        raise SystemExit(f"Unknown shock '{shock}'. Expected one of {shock_names}")
    shock_idx = int(shock_names.index(shock))

    var0 = float(QQ0[shock_idx, shock_idx])
    shock_delta = np.zeros((nshock,), dtype=float)
    shock_delta[shock_idx] = float(args.shock_size) * (np.sqrt(var0) if var0 > 0 else 1.0)

    cholQQ = _cov_factor_psd(QQ0)
    rng = np.random.default_rng(args.seed)

    girf_sum = np.zeros((args.h, len(obs_names)), dtype=float)
    k_base_sum = np.zeros((args.h, len(components)), dtype=float)
    k_shock_sum = np.zeros((args.h, len(components)), dtype=float)

    for _ in range(int(args.reps)):
        z = rng.standard_normal(size=(args.h, nshock))
        eps = z @ cholQQ.T

        base = _simulate_given_eps(model, params=params, eps_path=eps)
        shocked = _simulate_given_eps(model, params=params, eps_path=eps, shock_delta_t0=shock_delta)

        girf_sum += shocked["y_path"] - base["y_path"]
        if components:
            k_base_sum += base["s_path"]
            k_shock_sum += shocked["s_path"]

    girf = girf_sum / float(args.reps)

    out = pd.DataFrame(girf, columns=obs_names)
    if args.vars:
        want = [v.strip() for v in args.vars.split(",") if v.strip()]
        missing = sorted(set(want) - set(obs_names))
        if missing:
            raise SystemExit(f"--vars has unknown names: {missing}. Available: {obs_names}")
        out = out[want]

    print(f"GIRF (mean y_shocked - y_base), shock={shock}, size={args.shock_size}sd, reps={args.reps}")
    print(out.head(10).to_string(index=True))

    if components:
        k_base = pd.DataFrame(k_base_sum / float(args.reps), columns=components)
        k_shock = pd.DataFrame(k_shock_sum / float(args.reps), columns=components)
        print("\nMean chosen horizons (baseline), t=0..9")
        print(k_base.head(10).to_string(index=True))
        print("\nMean chosen horizons (shocked), t=0..9")
        print(k_shock.head(10).to_string(index=True))


if __name__ == "__main__":
    main()
