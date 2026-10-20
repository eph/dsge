# IRFOC

`dsge.irfoc.IRFOC` computes policy counterfactuals from a baseline forecast and responses to anticipated instrument shocks. The optimal-policy methods solve **perfect-foresight problems for one calibrated linear model**. They accept a string period loss, including the quadratic expressions used by `dsge.oc` and the squared shortfall and overshoot penalties described below.

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

The instrument shock supplies a response mapping; the optimal-policy methods do not require a chosen feedback rule. The baseline must include every variable used in the loss or constraints. The quadratic part of the loss must be convex. Model parameters are evaluated at `p0`. Use a model auxiliary such as `deli = i - i(-1)` for a rate-change penalty; direct leads and lags in the loss are unsupported.

## Commitment and discretion

```python
commitment = irfoc.simulate_optimal_control(loss, discount="beta")
discretion = irfoc.simulate_optimal_discretion(loss, discount="beta")
```

Commitment chooses the complete path at adoption. Discretion chooses the current action while accounting for how future policymakers reoptimize. Its continuation conditions follow the Hebden–Winkler construction; restarting a commitment problem every period would solve a different problem.

- The default `method="auto"` chooses `"stationary"` for a quadratic loss and `"finite"` when the loss contains a positive-weight squared shortfall or overshoot term. The selected method is recorded in `diagnostics["method"]`.
- `method="stationary"` iterates on time-invariant discretionary optimality conditions. `tolerance`, `max_iter`, and `damping` control the iteration. With inequalities, it uses the local construction described below and requires a slack terminal continuation.
- `method="finite"` works backward to a final policymaker. Loss constraints can bind through the final date. With inequalities, the solution is local to maintained future binding sets; it is not a global solution of a nonlinear discretionary game at switching boundaries.

Both truncate the response horizon. Compare early dates across increasing horizons and against the state-space solvers where available. A small numerical residual does not establish infinite-horizon determinacy.

Set the discount explicitly in comparisons: commitment retains the default `1.0`, while discretion defaults to `0.99`. Commitment's optional `ridge` and `u_weight` penalize implementing wedges and change the objective; both default to zero. Its legacy `dynamics="irf"` option remains an alias for the anticipated mapping, not discretion.

## Squared shortfall and overshoot losses

An output-shortfall penalty charges for negative output gaps and leaves positive gaps unpenalized:

```python
shortfall_loss = "pi**2 + 0.5*min(y, 0)**2 + deli**2"
shortfall_commitment = irfoc.simulate_optimal_control(
    shortfall_loss, discount="beta", return_details=True,
)
shortfall_discretion = irfoc.simulate_optimal_discretion(
    shortfall_loss, discount="beta", return_details=True,
)
```

The supported form is a convex quadratic remainder plus a sum of terms `c*min(a, 0)**2` or `c*max(a, 0)**2`, where `a` is affine in current variables and `c` is a nonnegative calibrated constant. For example, `max(pi - 0.02, 0)**2` penalizes inflation above a threshold. Both lowercase `min`/`max` and uppercase `Min`/`Max` are accepted. Zero-weight terms add no auxiliary variables.

Unsquared `min`/`max`, nested terms, products such as `pi*min(y, 0)`, negative penalty weights, and arbitrary `Piecewise` or indicator objectives are unsupported. Squaring a minimum of several affine expressions is also outside this grammar: each accepted minimum or maximum compares one affine expression with zero.

The compiler represents each penalty with an internal auxiliary choice and an affine inequality. For commitment this is an exact reformulation of the objective in the truncated problem, not a smooth approximation. Discretion uses the local active-set construction, with `"auto"` selecting the finite method so that an internal loss constraint may still bind at the last date. When comparing shortfall and symmetric discretion, set `method="finite"` for both if the comparison should keep the terminal convention unchanged.

The discretionary kernel also requires positive curvature in its augmented quadratic problem before imposing current inequalities. Some objectives made only of one-sided penalties fail this numerical requirement despite having a unique economic optimum. For example, write `y**2` directly instead of the equivalent `min(y, 0)**2 + max(y, 0)**2`. The solver raises an error rather than silently adding a penalty that changes the objective.

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

`constraints` accepts one inequality string or a list, using current-period variables and `>=` or `<=`. These explicit constraints can be combined with shortfall or overshoot losses. Commitment solves a constrained quadratic problem.

Constrained discretion holds future binding sets fixed when computing marginal effects. With `method="stationary"`, all constraints, including internal loss constraints, must be slack beyond `binding_horizon`; the unconstrained stationary solution supplies the continuation. With `method="finite"`, binding is permitted through the final date, and `binding_horizon` must be omitted. Corners where a marginal action changes future binding sets are outside either local construction. Failed iteration or residual checks raise errors.

These methods anticipate no additional future shocks. With binding constraints, they do not integrate the probability of shocks changing future binding sets.

## Results

The default return is an outcome `DataFrame`. With `return_details=True`, an `IRFOCOptimalResult` exposes:

| Attribute | Contents |
| --- | --- |
| `simulation` | Model outcomes and instrument levels. |
| `shocks` | Implementing instrument-shock wedges. |
| `residuals` | Policy-equation residuals, retaining the original instrument-column layout. |
| `multipliers` | Constraint multipliers in the undiscounted period convention. |
| `slack` | Nonnegative slack for feasible inequalities. |
| `diagnostics` | Regime, horizon, iterations, and numerical residuals; discretion also records the selected method and, when computed, its continuation residual. |

Internal loss auxiliaries do not change the economic columns of `simulation` or the instrument-shock columns of `shocks`. Their paths are available in `diagnostics["loss_auxiliaries"]`, a `DataFrame` whose columns name the canonical loss expressions. Internal constraint columns in `multipliers` and `slack` start with `"loss: "`, distinguishing them from explicit policy inequalities. These diagnostics may therefore be present even when `constraints=None`.

With one-sided losses, discretionary policy equations mix physical and auxiliary choices. `diagnostics["policy_system_residuals"]` exposes the complete augmented set of equations; the scalar `diagnostics["residual"]` always checks the full system, including constraints and auxiliary choices.

For a purely quadratic loss with no explicit constraints, `multipliers` and `slack` have no columns. Check optimality, feasibility, complementarity, and horizon sensitivity together.

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
uv run python dsge/examples/oc/nk_shortfall_demo.py
uv run python dsge/examples/oc/nk_shortfall_demo.py --compare-horizon 48
```

The sequence-policy example prints early-horizon differences from state-space solutions. The shortfall example compares symmetric and output-shortfall objectives under both policy arrangements, evaluates each path under both loss functions, and optionally compares two horizons. Its two discretionary runs both use the finite method. Existing examples remain in `dsge/examples/oc/nk_oc_demo.py` and `dsge/examples/irfoc/nk_zlb_rules_demo.py`.
