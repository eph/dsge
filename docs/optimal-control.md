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

The discretion method solves continuation optimality conditions following Hebden–Winkler. It does not select a simple feedback rule or restart commitment each period. Its default `method="auto"` selects stationary for a quadratic loss and finite for a loss with positive-weight squared shortfall or overshoot terms. The explicit stationary method above is appropriate for comparison with `compile_discretion`.

Compare solutions using the same loss, parameters, discount, shock, and inherited state. Sequence-space methods truncate responses, while the state-space benchmark has an infinite-horizon continuation. Compare early dates and increase the sequence horizon to assess terminal effects.

The sequence methods accept affine inequalities such as `constraints="i >= -0.2"`. Commitment solves a constrained quadratic problem. Constrained discretion uses a local active-set heuristic that holds future binding sets fixed when computing marginal effects. Its stationary method assumes a slack continuation after `binding_horizon`; its finite method permits binding at the final date. Neither covers every corner equilibrium. See [IRFOC](irfoc.md) for diagnostic fields and constraint assumptions.

## Output shortfalls and inflation overshoots

IRFOC additionally accepts nonnegative weighted squared penalties such as `min(y, 0)**2` and `max(pi - 0.02, 0)**2` alongside a convex quadratic remainder:

```python
shortfall_loss = "pi**2 + min(y, 0)**2 + deli**2"
shortfall_commitment = irfoc.simulate_optimal_control(
    shortfall_loss, discount="beta", return_details=True,
)
shortfall_discretion = irfoc.simulate_optimal_discretion(
    shortfall_loss, discount="beta", return_details=True,
)
```

Here commitment uses an exact auxiliary-variable reformulation of the finite-horizon objective. Discretion's default selects the finite local active-set method, allowing loss constraints to bind at the end. For a comparison that changes only the loss, use `method="finite"` for the symmetric discretionary problem too.

The affine argument may include calibrated targets. Unsquared or nested `min`/`max`, products with a shortfall term, negative weights, and arbitrary piecewise objectives are unsupported. This extension belongs to the IRFOC methods; the `dsge.oc` state-space compilers retain their quadratic-loss interface.

Returned economic paths and instrument-shock columns remain unchanged. With detailed results, `diagnostics["loss_auxiliaries"]` contains the internal penalty variables, and `"loss: "` prefixes identify their slack and multiplier columns. See [IRFOC](irfoc.md) for the complete grammar and limitations.

## Current scope and examples

This first sequence-space stage handles perfect foresight under one calibrated model. Forecast revisions, historical-data revisions, discrete model uncertainty, anticipated-utility learning, and tensor approximations remain later extensions. Rerunning a commitment calculation after new information does not by itself preserve earlier promises.

```bash
uv run python dsge/examples/oc/nk_sequence_policy_demo.py
uv run python dsge/examples/oc/nk_sequence_policy_demo.py --horizon 120
uv run python dsge/examples/oc/nk_shortfall_demo.py --compare-horizon 48
```

The sequence-policy example prints early-horizon differences from the state-space counterparts and numerical diagnostics. The shortfall example compares both objectives under commitment and discretion, with an optional longer-horizon check. Neither imposes a universal agreement threshold. The earlier `dsge/examples/oc/nk_oc_demo.py` provides a commitment comparison with optional plots.
