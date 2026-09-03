# Endogenous Horizons

FHP models can choose planning horizons endogenously through
`declarations.stopping_rule` (alias: `declarations.horizon_choice`). This turns
the YAML into an `EndogenousHorizonSwitchingModel`, so `read_yaml(...)` returns
the switching model directly and there is no separate `compile_model()` step.

Each component applies the incremental stopping rule

```text
k* = min { k >= 0 : MB(k + 1) < Δτ(k + 1) }, capped at k_max.
```

## Exponential planning costs

Each component can use its own cost schedule. Existing `cost: {a: ...}` models
retain flat marginal costs. For exponential marginal costs, write:

```yaml
cost:
  type: exponential
  a: household_cost / (beta*sigma)
  growth: cost_growth
lambda: 1 / (beta*sigma)
```

At added planning stage `j = k + 1`, the cost is
`Δτ(j) = a * exp(growth * (j - 1))`. Thus `a` is the cost of moving from horizon
zero to horizon one, and `growth` is a log growth rate: `growth: 0.31` multiplies
each successive marginal cost by `exp(0.31)`. Zero growth exactly recovers flat
costs. If total cost is instead specified as `B*(exp(b*k)-1)`, its marginal cost
corresponds to `a = B*(exp(b)-1)` and `growth = b`.

Both fields accept numbers or parameter-only expressions. They are recomputed
and cached for each parameter vector used by horizon selection, simulation,
GIRFs, and particle filtering. `a` must be finite and positive; `growth` must be
finite and nonnegative. The strict stopping comparison is unchanged: equality
continues planning. The finite search still ends at `k_max`; hitting that cap
does not establish that the unconstrained optimal horizon is finite.

The example `dsge/examples/fhp/nk_endogenous_exponential_costs.yaml` uses the
own-horizon GE projection with separate household and pricing objectives:

```python
from importlib.resources import files
import numpy as np
from dsge import read_yaml

path = files("dsge") / "examples/fhp/nk_endogenous_exponential_costs.yaml"
model = read_yaml(str(path))
params = np.array(model.p0, copy=True)
x = np.array([0.0025])  # 25 bp rate cut
model.choose_regime(params, x, t=0)  # (28, 59), growth = 0.31
params[model.parameter_names.index("cost_growth")] = 0.29
model.choose_regime(params, x, t=0)  # (59, 59)
```

The example sets shock persistence to one and caps both horizons at 59 to
represent the announcement impact of a 60-period peg. It does not encode the
peg's subsequent calendar-time exit. At this calibration the benefit-growth
threshold is approximately `0.30042`; the household stops at 28 above it.
Pricing reaches the example's cap in both cases. This is a linear-model
illustration, with very large responses at long horizons.

For linear marginal costs, `cost: {type: linear, a: ..., b: ...}` implements
`Δτ(j) = a + b*j` with `b >= 0`. Omitting `type` selects `linear`, and omitting
`b` sets it to zero. These schedules also work in `type: switching_ssm` YAML.
In the Python API, `cost_params` and `cost_func` accept
`ExponentialMarginalCostSchedule(a, growth)` or
`LinearMarginalCostSchedule(a, b)` objects; historical scalar and `(a, b)`
inputs remain supported. Import either schedule from `dsge`.

Costs that exceed floating-point range evaluate to positive infinity and stop
any finite marginal benefit. Nonfinite marginal benefits raise an error rather
than silently selecting a capped horizon.

## Beliefs about other components

The optional `belief_mode` determines which continuation a component uses when
horizons differ. The default is `correct`. It solves the joint mixed-horizon
economy along the saturated countdown
`(h, f) -> (max(h - 1, 0), max(f - 1, 0))`.

Set `belief_mode: own_horizon_projection` for the Woodford projection:

```yaml
stopping_rule:
  belief_mode: own_horizon_projection
  selection_mode: simultaneous
  components:
    household:
      k_max: 4
      assign_lhs: [y]
      cost: {a: household_cost}
      lambda: 1.0
      policy_object: y
    pricing:
      k_max: 4
      assign_lhs: [pi]
      cost: {a: pricing_cost}
      lambda: 1.0
      policy_object: pi
```

A component evaluating horizon `k` then solves a perceived economy in which
every planning row has horizon `k`, including rows with fixed actual horizons.
If realized horizons are `(h, f)`, the household rows use the diagonal continuation `(h - 1, h - 1)` and the pricing
rows use `(f - 1, f - 1)`. The realized current allocation still solves all
model equations jointly. Equal horizons nest the standard FHP solution exactly.

Horizon choices are independent of opponents' realized horizons under this
projection. Sequential and simultaneous selection therefore produce the same
profile away from ties, though simultaneous mode remains useful for explicit
diagnostics. Different components may have different `k_max` values; a component
can project its candidate `k` onto a component whose own choice grid ends sooner.

`own_horizon_projection` currently requires `declarations.expectations: 0`. A
single aggregate forecast row is not well-defined when row owners hold different
subjective forecast paths. Simulation, GIRFs, and particle filtering of current
observables remain supported.

## Sequential selection (default)

The backward-compatible default is `selection_mode: sequential`. Components
choose once in `selection_order`; a later component conditions on earlier
choices, and an earlier component is not revisited. If `selection_order` is
omitted, declaration order is used.

```yaml
stopping_rule:
  selection_mode: sequential       # optional; this is the default
  selection_order: [pricing, hh]
  components:
    # ...
```

## Simultaneous selection

Use `selection_mode: simultaneous` to select a mutual-best-response horizon
profile. For every point on the finite horizon grid, the model recomputes each
component's incremental-rule choice while holding every other component's
horizon fixed. A profile is admissible only if all of those best responses equal
the proposed coordinates.

```yaml
declarations:
  type: fhp
  stopping_rule:
    selection_mode: simultaneous
    equilibrium_selection: error
    components:
      household:
        k_max: 4
        assign_lhs: [c, q]
        cost: { a: 1e-4 }
        lambda: 1.0
        policy_object: "c + q"
      pricing:
        k_max: 10
        assign_lhs: [pi]
        cost: { a: 1e-4 }
        lambda: 1.0
        policy_object: "pi"
```

`selection_order` is not used in simultaneous mode. Enumeration, best-response
evaluation, and tie-breaking use a canonical component-name order, so changing
component declaration order or `selection_order` does not change the selected
component-to-horizon mapping.

There can be zero, one, or several pure fixed points:

- One profile is returned directly.
- No profile raises `NoPureHorizonEquilibriumError`. The exception carries a
  `.diagnostics` object with every candidate and best-response profile, and its
  message reports the closest candidates and deviating components.
- Several profiles raise `MultipleHorizonEquilibriaError` by default and list
  the equilibria. This avoids silently adding an economic selection assumption.
  Set `equilibrium_selection` explicitly to `lexicographic_min` or
  `lexicographic_max` to select by horizon tuples ordered by sorted component
  name. These policies are deterministic but are modeling choices.

The finite-grid implementation caches a component's best response by the other
coordinates during each selection. Thus, a component response is evaluated once
per distinct opponents' profile rather than once per value of its own proposed
horizon.

## Lower-level Python API

The constructor accepts the same configuration directly:

```python
from dsge import EndogenousHorizonSwitchingModel

model = EndogenousHorizonSwitchingModel(
    # existing components, k_max, costs, lambda, solution and policy hooks ...
    selection_mode="simultaneous",
    equilibrium_selection="error",
)
```

The fixed-point machinery is also available without changing the model's active
selection mode:

```python
br = model.component_best_response(
    params, x_t, t=0,
    component="pricing",
    other_horizons={"household": 2},
)
equilibria = model.find_simultaneous_regimes(params, x_t, t=0)
diagnostics = model.simultaneous_regime_diagnostics(params, x_t, t=0)
```

`other_horizons` must contain every other component and omit the component being
optimized. Custom `info_func` and `mb_func` receive that complete, canonically
ordered mapping. Custom marginal benefits, `policy_object`, and parameter-driven
cost/lambda functions therefore work in both modes.

## Simulation, GIRFs, and filtering

```python
from dsge import read_yaml

m = read_yaml("dsge/examples/fhp/partial_equilibrium_endogenous.yaml")
sim = m.simulate(params=m.p0, T=200, seed=123)
girf = m.girf(m.p0, shock="e_y", h=20, reps=200, seed=123)
loglik, stats = m.pf_loglik(m.p0, data, nparticles=2000, seed=123)
```

All downstream paths call `choose_regime(...)`, so the configured selection mode
is used consistently. GIRF output includes mean baseline and shocked horizons.

## Examples

- One-component sequential YAML:
  `dsge/examples/fhp/partial_equilibrium_endogenous.yaml`
- Two-component simultaneous YAML (household 0..4, pricing 0..10):
  `dsge/examples/fhp/fhp_endogenous_two_component.yaml`
- Two-component Woodford-projection NK YAML:
  `dsge/examples/fhp/nk_endogenous_own_horizon_projection.yaml`
- Monte Carlo GIRF script: `dsge/examples/fhp/girf_endogenous_horizon.py`

Run the GIRF example with:

```bash
uv run python dsge/examples/fhp/girf_endogenous_horizon.py
```

## Fixed horizons with own-horizon projection

The same belief construction is available directly when compiling a fixed FHP
model:

```python
linear = fhp_model.compile_model(
    k={"default": 2, "by_lhs": {"c": 1, "q": 1, "pi": 2}},
    belief_mode="own_horizon_projection",
)
```

Here each row uses the continuation from the common-horizon economy associated
with its own remaining horizon. Rows whose horizon is zero use their terminal
equation. The default `belief_mode="correct"` retains the joint saturated
countdown used by earlier versions.
