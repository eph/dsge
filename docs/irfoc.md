# IRFOC

`dsge.irfoc.IRFOC` computes policy counterfactuals from a baseline forecast and responses to anticipated instrument shocks. The optimal-policy methods solve **perfect-foresight problems for one calibrated linear model**, using the same string period-loss format as `dsge.oc`.

## Set up a policy problem

```python
from dsge import read_yaml
from dsge.irfoc import IRFOC

model = read_yaml("dsge/examples/oc/nk_oc_demo.yaml")
p0 = model.p0()
compiled = model.compile_model()
columns = ["pi", "y", "i", "u", "re", "deli"]
baseline = compiled.impulse_response(p0, h=80)["er"].loc[:, columns]
irfoc = IRFOC(
    model, baseline=baseline, instrument_shocks="em", p0=p0,
    compiled_model=compiled,
)
loss = "pi**2 + y**2 + deli**2"
```

The instrument shock supplies a response mapping; the optimal-policy methods do not require a chosen feedback rule. The baseline must include every variable used in the loss or constraints. Losses must be convex quadratic polynomials in current model variables. Model parameters are evaluated at `p0`. Use a model auxiliary such as `deli = i - i(-1)` for a rate-change penalty; direct leads and lags in the loss are unsupported.

## Commitment and discretion

```python
commitment = irfoc.simulate_optimal_control(loss, discount="beta")
discretion = irfoc.simulate_optimal_discretion(
    loss, discount="beta", method="stationary",
)
```

Commitment chooses the complete path at adoption. Discretion chooses the current action while accounting for how future policymakers reoptimize. Its continuation conditions follow the Hebden–Winkler construction; restarting a commitment problem every period would solve a different problem.

- `method="stationary"` iterates on time-invariant discretionary optimality conditions. `tolerance`, `max_iter`, and `damping` control the iteration.
- `method="finite"` solves a finite game backward, with a different terminal continuation.

Both truncate the response horizon. Compare early dates across increasing horizons and against the state-space solvers where available. A small numerical residual does not establish infinite-horizon determinacy.

Set the discount explicitly in comparisons: commitment retains the default `1.0`, while discretion defaults to `0.99`. Commitment's optional `ridge` and `u_weight` penalize implementing wedges and change the objective; both default to zero. Its legacy `dynamics="irf"` option remains an alias for the anticipated mapping, not discretion.

## Affine inequality constraints

```python
bounded_commitment = irfoc.simulate_optimal_control(
    loss, discount="beta", constraints="i >= -0.2", return_details=True,
)
bounded_discretion = irfoc.simulate_optimal_discretion(
    loss, discount="beta", method="stationary",
    constraints="i >= -0.2", binding_horizon=20, return_details=True,
)
```

`constraints` accepts one inequality string or a list, using current-period variables and `>=` or `<=`. Commitment solves a constrained quadratic problem. Constrained discretion uses a local active-set heuristic: future binding sets are fixed when computing marginal effects, and constraints must be slack beyond `binding_horizon`. It requires `method="stationary"` to supply the unconstrained continuation. Corners where a marginal action changes future binding sets are outside this construction. Failed iteration or residual checks raise errors.

These methods anticipate no additional future shocks. With binding constraints, they do not integrate the probability of shocks changing future binding sets. The loss parser does not directly accept `min`, `max`, or piecewise objectives.

## Results

The default return is an outcome `DataFrame`. With `return_details=True`, an `IRFOCOptimalResult` exposes:

| Attribute | Contents |
| --- | --- |
| `simulation` | Model outcomes and instrument levels. |
| `shocks` | Implementing instrument-shock wedges. |
| `residuals` | Optimality-equation residuals. |
| `multipliers` | Constraint multipliers in the undiscounted period convention. |
| `slack` | Nonnegative slack for feasible inequalities. |
| `diagnostics` | Regime, horizon, iterations, and numerical residuals; discretion also reports its method and continuation residual. |

Without constraints, `multipliers` and `slack` have no columns. Check optimality, feasibility, complementarity, and horizon sensitivity together.

## Existing instrument-rule interfaces

The existing methods remain available for evaluating specified instrument equations:

```python
counterfactual = irfoc.simulate("i = 0.85*i(-1) + 0.15*(1.5*pi + 0.5*y)")
bounded_rule = irfoc.simulate_piecewise(
    "i = max(-0.2, 1.5*pi + 0.5*y)", u_bounds=(-5.0, 5.0),
)
```

`simulate_piecewise` uses the MILP backend. Its `u_bounds` constrain implementing wedges and must contain the required solution. These methods enforce specified equations; the optimal-policy methods derive actions from the loss.

## Scope and examples

This first stage does not implement forecast-vintage updates that preserve past commitments, historical data revisions, uncertain transmission, anticipated-utility learning, or tensor approximations. Those are separate extensions, not capabilities implied by the current methods.

Algorithm references are Sections 3.2, 3.3, 5.2, and 5.3 of the [August 2024 FEDS revision of Hebden and Winkler](https://www.federalreserve.gov/econres/feds/files/2021042r1pap.pdf); section numbering differs in the published article. The present implementation does not claim coverage of every case in the paper.

Run the comparison of sequence-space and state-space commitment and discretion:

```bash
uv run python dsge/examples/oc/nk_sequence_policy_demo.py
uv run python dsge/examples/oc/nk_sequence_policy_demo.py --horizon 120
```

The example prints early-horizon differences and diagnostics using the same model, shock, loss, and discount. Existing examples remain in `dsge/examples/oc/nk_oc_demo.py` and `dsge/examples/irfoc/nk_zlb_rules_demo.py`.
