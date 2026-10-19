# Optimal Control

DSGE computes optimal policy either from structural state-space equations (`dsge.oc`) or from forecasts and anticipated instrument responses (`dsge.irfoc.IRFOC`). Both accept a string describing the period loss.

## State-space solutions

```python
from dsge import read_yaml
from dsge.oc import compile_commitment, compile_discretion

model = read_yaml("dsge/examples/oc/nk_oc_demo.yaml")
p0 = model.p0()
loss = "pi**2 + y**2 + deli**2"
commitment = compile_commitment(
    model, loss, policy_instruments="i", policy_shocks="em", beta="beta",
)
discretion = compile_discretion(
    model, loss, policy_instruments="i", policy_shocks="em", beta="beta",
)
commitment_irf = commitment.impulse_response(p0, h=80)["er"]
discretion_irf = discretion.impulse_response(p0, h=80)["er"]
```

`policy_instruments` identifies the controlled variables; `policy_shocks` identifies the corresponding shocks in the baseline equations. `beta` accepts a number or model parameter name. The model variable `deli = i - i(-1)` supplies the rate-change penalty.

Commitment chooses current and future actions together. Under discretion, future policymakers reoptimize and the current policymaker anticipates their responses. These are distinct problems with the same preferences and model parameters.

## Perfect-foresight sequence-space solutions

```python
from dsge.irfoc import IRFOC

baseline_model = model.compile_model()
columns = ["pi", "y", "i", "u", "re", "deli"]
baseline = baseline_model.impulse_response(p0, h=80)["er"].loc[:, columns]
irfoc = IRFOC(
    model, baseline=baseline, instrument_shocks="em", p0=p0,
    compiled_model=baseline_model,
)
commitment_path = irfoc.simulate_optimal_control(loss, discount="beta")
discretion_path = irfoc.simulate_optimal_discretion(
    loss, discount="beta", method="stationary",
)
```

The discretion method solves continuation optimality conditions following Hebden–Winkler. It does not select a simple feedback rule or restart commitment each period. `method="finite"` instead solves a finite backward game; use the stationary method for comparison with `compile_discretion`.

Compare solutions using the same loss, parameters, discount, shock, and inherited state. Sequence-space methods truncate responses, while the state-space benchmark has an infinite-horizon continuation. Compare early dates and increase the sequence horizon to assess terminal effects.

The sequence methods accept affine inequalities such as `constraints="i >= -0.2"`. Commitment solves a constrained quadratic problem. Constrained discretion requires the stationary method and uses a local active-set heuristic with an unconstrained tail, rather than covering every corner equilibrium. See [IRFOC](irfoc.md) for `binding_horizon`, result diagnostics, and constraint assumptions.

## Current scope and examples

This first sequence-space stage handles perfect foresight under one calibrated model. Forecast revisions, historical-data revisions, discrete model uncertainty, anticipated-utility learning, and tensor approximations remain later extensions. Rerunning a commitment calculation after new information does not by itself preserve earlier promises.

```bash
uv run python dsge/examples/oc/nk_sequence_policy_demo.py
uv run python dsge/examples/oc/nk_sequence_policy_demo.py --horizon 120
```

The example prints both policy paths, their early-horizon differences from the state-space counterparts, and numerical diagnostics. It imposes no universal agreement threshold. The earlier `dsge/examples/oc/nk_oc_demo.py` provides a commitment comparison with optional plots.
