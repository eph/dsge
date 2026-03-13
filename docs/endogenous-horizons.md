# Endogenous Horizons

FHP models can choose planning horizons endogenously through `declarations.stopping_rule` (alias: `declarations.horizon_choice`).

This turns the YAML into an `EndogenousHorizonSwitchingModel`, so the workflow is different from a fixed-horizon `FHPRepAgent`:

- `read_yaml(...)` returns a switching model directly
- there is no `compile_model()` step
- you use `simulate(...)`, `girf(...)`, `choose_regime(...)`, and `pf_loglik(...)`

## YAML sketch

```yaml
declarations:
  type: fhp
  stopping_rule:
    components:
      pricing:
        k_max: 8
        assign_lhs: [pi]
        cost: { a: 1e-4 }
        lambda: "(-D_pp)/(1-beta*theta)"
        policy_object: "theta/(1-theta) * pi"
```

The stopping rule chooses a discrete horizon each period based on the current reduced state and the configured `policy_object`.

## Simulation

```python
from dsge import read_yaml

m = read_yaml("dsge/examples/fhp/partial_equilibrium_endogenous.yaml")
sim = m.simulate(params=m.p0, T=200, seed=123)
girf = m.girf(m.p0, shock="e_y", h=20, reps=200, seed=123)
```

The GIRF output contains both observables and horizon-choice summaries such as mean chosen horizons in the baseline and shocked simulations.

## Examples

- YAML: `dsge/examples/fhp/partial_equilibrium_endogenous.yaml`
- Monte Carlo GIRF script: `dsge/examples/fhp/girf_endogenous_horizon.py`
- Model-family overview: `docs/models.md`

Run the GIRF example:

```bash
uv run python dsge/examples/fhp/girf_endogenous_horizon.py
```
