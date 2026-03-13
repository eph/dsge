# IRFOC

`dsge.irfoc.IRFOC` solves finite-horizon counterfactual paths for a linear model by choosing an instrument-shock sequence that enforces a policy rule along a baseline path.

Typical workflow:

```python
from dsge import read_yaml
from dsge.irfoc import IRFOC

m = read_yaml("dsge/examples/irfoc/nk_irfoc_demo.yaml")
p0 = m.p0()
lin = m.compile_model()

baseline = lin.impulse_response(p0, h=40)["er"].loc[:, ["pi", "y", "i", "re", "u", "deli"]]
irfoc = IRFOC(m, baseline=baseline, instrument_shocks="em", p0=p0, compiled_model=lin)

cf = irfoc.simulate("i = 1.5*pi + 0.5*y")
```

## Affine rules

Use `simulate(...)` for affine rules in current and lagged variables:

- `i = 1.5*pi + 0.5*y`
- `i = 0.85*i(-1) + 0.15*(1.5*pi + 0.5*y)`

If you need the implied instrument-shock path, pass `return_details=True` and inspect the returned `IRFOCResult`.

## Piecewise rules

Rules containing `max()` or `min()` use the MILP backend via `simulate_piecewise(...)`:

```python
cf = irfoc.simulate_piecewise(
    "i = max(-0.2, 1.5*pi + 0.5*y)",
    u_bounds=(-5.0, 5.0),
)
```

`u_bounds` should be wide enough to contain the implied instrument-shock sequence. The example script below computes robust bounds from unconstrained runs first.

## Examples

- Script: `dsge/examples/irfoc/nk_zlb_rules_demo.py`
- Model: `dsge/examples/irfoc/nk_irfoc_demo.yaml`
- Example notes: `dsge/examples/irfoc/README.md`

Run the bundled demo:

```bash
uv run python dsge/examples/irfoc/nk_zlb_rules_demo.py
```

It compares several Taylor-style rules under a ZLB implemented as `max(zlb, rule)` and writes CSV/PNG outputs under `dsge/examples/irfoc/_out/`.
