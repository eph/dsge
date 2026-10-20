"""Compare symmetric and output-shortfall losses under perfect foresight.

Run from the repository root:
    uv run python dsge/examples/oc/nk_shortfall_demo.py
    uv run python dsge/examples/oc/nk_shortfall_demo.py --compare-horizon 48

Both discretionary comparisons use the finite method so only the loss changes,
not the terminal convention. Its active-set solution is local to the maintained
future binding sets, rather than a global nonlinear discretionary equilibrium.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dsge import read_yaml
from dsge.irfoc import IRFOC


LOSSES = {
    "symmetric": "pi**2 + y**2 + deli**2",
    "shortfall": "pi**2 + min(y, 0)**2 + deli**2",
}
COLUMNS = ["pi", "y", "i", "u", "re", "deli"]


def evaluate_losses(path: pd.DataFrame, beta: float) -> tuple[float, float]:
    """Evaluate the original economic losses, not auxiliary objectives."""
    weights = beta ** np.arange(len(path))
    common = path["pi"].to_numpy() ** 2 + path["deli"].to_numpy() ** 2
    output = path["y"].to_numpy()
    symmetric = weights @ (common + output ** 2)
    shortfall = weights @ (common + np.minimum(output, 0.0) ** 2)
    return float(symmetric), float(shortfall)


def solve_at_horizon(model, compiled, p0, horizon: int):
    baseline = compiled.impulse_response(p0, h=horizon)["er"].loc[:, COLUMNS]
    irfoc = IRFOC(
        model, baseline=baseline, instrument_shocks="em", p0=p0,
        compiled_model=compiled,
    )
    beta = float(p0[[str(p) for p in model["parameters"]].index("beta")])
    rows, paths = [], {}
    for regime in ("commitment", "discretion"):
        for name, loss in LOSSES.items():
            if regime == "commitment":
                result = irfoc.simulate_optimal_control(
                    loss, discount="beta", return_details=True,
                )
            else:
                result = irfoc.simulate_optimal_discretion(
                    loss, discount="beta", method="finite", return_details=True,
                )
            path = result.simulation
            symmetric, shortfall = evaluate_losses(path, beta)
            rows.append({
                "horizon": horizon,
                "regime": regime,
                "objective": name,
                "initial rate": float(path["i"].iloc[0]),
                "initial inflation": float(path["pi"].iloc[0]),
                "initial output gap": float(path["y"].iloc[0]),
                "symmetric loss": symmetric,
                "shortfall loss": shortfall,
                "residual": float(result.diagnostics["residual"]),
            })
            paths[(regime, name)] = path.loc[:, ["i", "pi", "y"]]
    return rows, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--compare-horizon", type=int, default=None)
    args = parser.parse_args()
    if args.horizon < 2:
        parser.error("horizon must be at least 2")
    if args.compare_horizon is not None and args.compare_horizon <= args.horizon:
        parser.error("compare-horizon must exceed horizon")

    model = read_yaml(str(Path(__file__).with_name("nk_oc_demo.yaml")))
    p0 = model.p0()
    compiled = model.compile_model()
    rows, paths = solve_at_horizon(model, compiled, p0, args.horizon)
    comparison = []
    if args.compare_horizon is not None:
        more_rows, longer_paths = solve_at_horizon(
            model, compiled, p0, args.compare_horizon,
        )
        rows.extend(more_rows)
        periods = min(10, args.horizon + 1)
        for (regime, objective), path in paths.items():
            gap = (
                path.iloc[:periods].to_numpy()
                - longer_paths[(regime, objective)].iloc[:periods].to_numpy()
            )
            comparison.append({
                "regime": regime,
                "objective": objective,
                "dates compared": periods,
                "max path difference": float(np.max(np.abs(gap))),
            })

    print("Positive natural-rate shock; discount beta from nk_oc_demo.yaml.")
    print("Symmetric loss:", LOSSES["symmetric"])
    print("Shortfall loss:", LOSSES["shortfall"])
    print("Both discretion runs use method='finite'.")
    print(pd.DataFrame(rows).to_string(
        index=False, float_format=lambda x: f"{x:.6g}",
    ))
    print("\nInitial paths at the first horizon:")
    print(pd.concat(paths, axis=1).head(5).to_string(
        float_format=lambda x: f"{x:.5f}",
    ))
    if comparison:
        print("\nSensitivity to the longer horizon (same objective and regime):")
        print(pd.DataFrame(comparison).to_string(
            index=False, float_format=lambda x: f"{x:.6g}",
        ))


if __name__ == "__main__":
    main()
