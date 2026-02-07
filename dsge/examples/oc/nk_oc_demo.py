from __future__ import annotations

from pathlib import Path

import numpy as np

from dsge import read_yaml
from dsge.irfoc import IRFOC
from dsge.oc import compile_commitment


def main() -> None:
    here = Path(__file__).resolve().parent
    model = read_yaml(str(here / "nk_oc_demo.yaml"))
    p0 = model.p0()

    # Finite-horizon IRFOC approximates the infinite-horizon commitment policy when the horizon is
    # long enough. We solve on a long horizon and only plot / compare the early part.
    h_plot = 80
    h_full = 250

    # Baseline (Taylor-rule) compilation + baseline IRF.
    mod_base = model.compile_model()
    cols = ["pi", "y", "i", "u", "re", "deli"]
    baseline = mod_base.impulse_response(p0, h=h_full)["er"].loc[:, cols]

    # Commitment optimal policy (Dennis-style LRE + KKT system).
    loss = "pi**2 + y**2 + deli**2"
    mod_commit = compile_commitment(model, loss, policy_instruments="i", policy_shocks="em", beta="beta")
    irf_commit = mod_commit.impulse_response(p0, h=h_full)["er"].loc[:, cols]

    # Optional cross-check: IRFOC quadratic problem should coincide with OC commitment (same loss).
    irfoc = IRFOC(model, baseline=baseline, instrument_shocks="em", p0=p0, compiled_model=mod_base)
    irf_irfoc = irfoc.simulate_optimal_control(loss, discount="beta").loc[:, cols]

    diff = (irf_commit.iloc[: h_plot + 1] - irf_irfoc.iloc[: h_plot + 1]).to_numpy()
    max_abs = float(np.max(np.abs(diff)))
    print(f"OC commitment vs IRFOC(optimal_control) max|diff| over 0..{h_plot}: {max_abs:.3g}")

    out_dir = here / "_out"
    out_dir.mkdir(exist_ok=True)
    irf_commit.to_csv(out_dir / "commitment_oc.csv", index=True)
    irf_irfoc.to_csv(out_dir / "irfoc_quadratic.csv", index=True)
    baseline.to_csv(out_dir / "baseline_taylor.csv", index=True)
    print(f"Wrote CSVs to: {out_dir}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot.")
        return

    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for label, df in [("baseline", baseline), ("commitment", irf_commit)]:
        dfp = df.iloc[: h_plot + 1]
        ax[0].plot(dfp.index, dfp["i"], label=label)
        ax[1].plot(dfp.index, dfp["pi"], label=label)
        ax[2].plot(dfp.index, dfp["y"], label=label)

    ax[0].set_ylabel("i")
    ax[1].set_ylabel("pi")
    ax[2].set_ylabel("y")
    ax[2].set_xlabel("Horizon")
    ax[0].legend()

    fig.suptitle("NK example: baseline vs OC commitment")
    fig.tight_layout()
    out_png = out_dir / "nk_oc_demo.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote plot to: {out_png}")


if __name__ == "__main__":
    main()
