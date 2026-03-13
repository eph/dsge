# Optimal Control

The `dsge.oc` module provides Dennis-style optimal-control solvers for linear DSGE models. The main public entry points are `compile_commitment(...)` and `compile_discretion(...)`.

## Commitment

```python
from dsge import read_yaml
from dsge.oc import compile_commitment

m = read_yaml("dsge/examples/oc/nk_oc_demo.yaml")
p0 = m.p0()

loss = "pi**2 + y**2 + deli**2"
mod = compile_commitment(
    m,
    loss,
    policy_instruments="i",
    policy_shocks="em",
    beta="beta",
)

irf = mod.impulse_response(p0, h=80)["er"]
```

Arguments:

- `loss`: quadratic loss string in model variables
- `policy_instruments`: instrument variable(s) the planner controls
- `policy_shocks`: shock(s) used to implement the instrument path
- `beta`: discount factor or parameter name

## Discretion

`compile_discretion(...)` uses the same interface for discretionary policy problems:

```python
from dsge.oc import compile_discretion

mod = compile_discretion(
    m,
    loss,
    policy_instruments="i",
    policy_shocks="em",
    beta="beta",
)
```

## Relation To IRFOC

For long enough horizons, `IRFOC.simulate_optimal_control(...)` provides a useful finite-horizon cross-check for commitment problems with the same quadratic loss.

The bundled demo does exactly that:

- Script: `dsge/examples/oc/nk_oc_demo.py`
- Model: `dsge/examples/oc/nk_oc_demo.yaml`
- Example notes: `dsge/examples/oc/README.md`

Run it with:

```bash
uv run python dsge/examples/oc/nk_oc_demo.py
```
