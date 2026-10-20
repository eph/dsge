"""Compare perfect-foresight sequence-space and state-space optimal policy.

Run from the repository root with:
    uv run python dsge/examples/oc/nk_sequence_policy_demo.py --horizon 80

Increase the horizon and compare early dates to assess truncation effects.
Reported gaps are diagnostics, not an assertion of exact finite-horizon
agreement with infinite-horizon state-space solutions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dsge import read_yaml
from dsge.irfoc import IRFOC
from dsge.oc import compile_commitment, compile_discretion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--compare-periods", type=int, default=20)
    args = parser.parse_args()
    if args.horizon < 2 or not 1 <= args.compare_periods < args.horizon:
        parser.error("require horizon >= 2 and 1 <= compare-periods < horizon")

    model = read_yaml(str(Path(__file__).with_name("nk_oc_demo.yaml")))
    p0 = model.p0()
    loss = "pi**2 + y**2 + deli**2"
    columns = ["pi", "y", "i", "u", "re", "deli"]
    shock = "er"
    baseline_model = model.compile_model()
    baseline = baseline_model.impulse_response(p0, h=args.horizon)[shock]
    baseline = baseline.loc[:, columns]
    irfoc = IRFOC(
        model, baseline=baseline, instrument_shocks="em", p0=p0,
        compiled_model=baseline_model,
    )

    sequence = {
        "commitment": irfoc.simulate_optimal_control(
            loss, discount="beta", return_details=True,
        ),
        "discretion": irfoc.simulate_optimal_discretion(
            loss, discount="beta", method="stationary", return_details=True,
        ),
    }
    compilers = {
        "commitment": compile_commitment,
        "discretion": compile_discretion,
    }
    comparisons = []
    paths = {}
    for regime, compile_policy in compilers.items():
        reference_model = compile_policy(
            model, loss, policy_instruments="i", policy_shocks="em", beta="beta",
        )
        reference = reference_model.impulse_response(p0, h=args.horizon)[shock]
        reference = reference.loc[:, columns]
        result = sequence[regime]
        difference = (
            result.simulation.iloc[:args.compare_periods].to_numpy()
            - reference.iloc[:args.compare_periods].to_numpy()
        )
        comparisons.append({
            "policy": regime,
            "max early gap": float(np.max(np.abs(difference))),
            "optimality residual": float(np.max(np.abs(result.residuals.to_numpy()))),
            "solver residual": float(result.diagnostics["residual"]),
            "iterations": result.diagnostics["iterations"],
        })
        paths[f"{regime}: sequence"] = result.simulation.loc[:, ["i", "pi", "y"]]
        paths[f"{regime}: state space"] = reference.loc[:, ["i", "pi", "y"]]

    print(f"Period loss: {loss}; discount: beta; shock: {shock}")
    print(
        f"Sequence horizon: {len(baseline)} dates; "
        f"comparison: first {args.compare_periods} dates"
    )
    print(
        pd.DataFrame(comparisons).set_index("policy").to_string(
            float_format=lambda x: f"{x:.3e}",
        )
    )
    print("\nInitial policy paths:")
    print(pd.concat(paths, axis=1).head(5).to_string(
        float_format=lambda x: f"{x:.5f}",
    ))
    print(
        "\nIncrease --horizon to assess the terminal approximation; "
        "no future forecast revisions are simulated."
    )


if __name__ == "__main__":
    main()
