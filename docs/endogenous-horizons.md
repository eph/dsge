# Endogenous Horizons

FHP models can choose planning horizons endogenously through
`declarations.stopping_rule` (alias: `declarations.horizon_choice`). This turns
the YAML into an `EndogenousHorizonSwitchingModel`, so `read_yaml(...)` returns
the switching model directly and there is no separate `compile_model()` step.

Each component applies the incremental stopping rule

```text
k* = min { k >= 0 : MB(k + 1) < Δτ(k + 1) }, capped at k_max.
```

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
- Monte Carlo GIRF script: `dsge/examples/fhp/girf_endogenous_horizon.py`

Run the GIRF example with:

```bash
uv run python dsge/examples/fhp/girf_endogenous_horizon.py
```
