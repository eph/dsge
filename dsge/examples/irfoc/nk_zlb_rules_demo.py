from __future__ import annotations

from pathlib import Path

import numpy as np

from dsge import read_yaml
from dsge.irfoc import IRFOC


def main() -> None:
    here = Path(__file__).resolve().parent
    model_path = here / "nk_irfoc_demo.yaml"

    model = read_yaml(str(model_path))
    p0 = model.p0()

    # Horizon for the perfect-foresight rule comparison.
    T = 40

    # A large negative demand shock (implemented as a negative natural-rate shock).
    shock_scale = -1.2

    # Simple Taylor(1999)-style coefficients.
    phi_pi = 1.5
    phi_y = 0.5

    # Note: in this linearized NK demo, `i` is a deviation from steady state, so a level ZLB at
    # i_level >= 0 corresponds to i_deviation >= -i_ss. We pick a (toy) steady-state nominal rate
    # i_ss and enforce the ZLB as `max(-i_ss, rule)`; we plot i_level = i_deviation + i_ss.
    i_ss = 0.2
    zlb = -i_ss

    compiled = model.compile_model()
    cols = ["pi", "y", "i", "re", "u", "deli"]
    baseline = shock_scale * compiled.impulse_response(p0, h=T)["er"].loc[:, cols]

    irfoc = IRFOC(model, baseline=baseline, instrument_shocks="em", p0=p0, compiled_model=compiled)

    rules = {
        "Taylor99": f"i = max({zlb}, {phi_pi}*pi + {phi_y}*y)",
        "InertialTaylor99": f"i = max({zlb}, 0.85*i(-1) + 0.15*({phi_pi}*pi + {phi_y}*y))",
        "FirstDiffRule": f"i = max({zlb}, i(-1) + {phi_pi}*pi + {phi_y}*y)",
    }

    # Piecewise rules require the MILP backend. We pick robust bounds for the instrument shock by
    # first solving the *unconstrained* (no max) versions of the rules and inflating their implied
    # MP-wedge magnitudes.
    unconstrained_rules = [
        f"i = {phi_pi}*pi + {phi_y}*y",
        f"i = 0.85*i(-1) + 0.15*({phi_pi}*pi + {phi_y}*y)",
        f"i = i(-1) + {phi_pi}*pi + {phi_y}*y",
    ]
    max_abs_em = 0.0
    for r in unconstrained_rules:
        res = irfoc.simulate(r, return_details=True)
        max_abs_em = max(max_abs_em, float(res.shocks.abs().to_numpy().max()))
    u_bound = 25.0 * max(1.0, max_abs_em)
    u_bounds = (-u_bound, u_bound)

    sims = {}
    for name, rule in rules.items():
        df = irfoc.simulate_piecewise(rule, u_bounds=u_bounds).loc[:, ["i", "pi"]]
        df = df.assign(ffr=(df["i"] + i_ss))
        sims[name] = df

    out_dir = here / "_out"
    out_dir.mkdir(exist_ok=True)
    for name, df in sims.items():
        df.to_csv(out_dir / f"{name}.csv", index=True)

    bind_counts = {
        name: int(np.sum(np.isclose(df["i"].to_numpy(), zlb, atol=1e-8)))
        for name, df in sims.items()
    }
    print(f"ZLB (i deviation) = {zlb:.3g}; i_ss = {i_ss:.3g}; shock_scale={shock_scale}")
    print(f"Implied MILP instrument-shock bounds: {u_bounds}")
    print(f"ZLB binding periods: {bind_counts}")
    print(f"Wrote CSVs to: {out_dir}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot.")
        return

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for name, df in sims.items():
        ax[0].plot(df.index, df["ffr"], label=name)
        ax[1].plot(df.index, df["pi"], label=name)

    ax[0].axhline(0.0, color="k", linewidth=1)
    ax[0].set_ylabel("FFR (level)")
    ax[0].legend()

    ax[1].set_ylabel("Inflation / pi")
    ax[1].set_xlabel("Horizon")

    fig.suptitle("Simple NK: ZLB rule comparison (IRFOC)")
    fig.tight_layout()
    out_png = out_dir / "nk_zlb_rules_demo.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote plot to: {out_png}")


if __name__ == "__main__":
    main()
