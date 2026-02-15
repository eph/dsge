#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from dsge import read_yaml


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

    obs_subset = None
    if args.vars:
        obs_subset = [v.strip() for v in str(args.vars).split(",") if v.strip()]

    out: dict[str, Any] = model.girf(
        params,
        shock=args.shock,
        h=int(args.h),
        reps=int(args.reps),
        shock_size=float(args.shock_size),
        seed=int(args.seed),
        obs_subset=obs_subset,
    )

    print(
        f"GIRF (mean y_shocked - y_base), shock={out['shock']}, size={args.shock_size}sd, reps={args.reps}"
    )
    print(out["girf"].head(10).to_string(index=True))
    print("\nMean chosen horizons (baseline), t=0..9")
    print(out["k_base_mean"].head(10).to_string(index=True))
    print("\nMean chosen horizons (shocked), t=0..9")
    print(out["k_shocked_mean"].head(10).to_string(index=True))


if __name__ == "__main__":
    main()
