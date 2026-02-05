# Second-Order Perturbation (LRE)

This project supports **second-order perturbation** for standard **linear rational expectations (LRE)** DSGE models.

At a high level:

- `order=1` uses the existing **linear-Gaussian** state-space representation (Kalman filter).
- `order=2` uses a **pruned second-order decision rule** (Dynare-style) and a **particle-filter** likelihood.

## Solve Second Order Directly

```python
from dsge import read_yaml

m = read_yaml("dsge/examples/nkmp/nkmp.yaml")
p0 = m.p0()

sol = m.solve_second_order(p0)
print(sol.gx.shape, sol.gxx.shape)
```

`solve_second_order()` returns a `SecondOrderSolution` with first- and second-order coefficients. You can also export a
Dynare-like array layout for direct comparison:

```python
d = sol.as_dynare_like()
print(d["ghx"].shape, d["ghxx"].shape)
```

## Compile an Order-2 Model (Particle Filter)

```python
c = m.compile_model(order=2, pruning=True)
ll = c.log_lik(p0, nparticles=2000, seed=0)
```

Particle-filter tuning lives on the likelihood call:

- `nparticles` (int): number of particles (default is model-dependent; typically a few thousand)
- `seed` (int | None): RNG seed for reproducibility
- `resample_threshold` (float in `(0, 1]`): effective-sample-size threshold

The compiled object also supports simulation and IRFs:

```python
irfs = c.impulse_response(p0, h=20)
sim = c.simulate(p0, nsim=200, seed=0)
```

## Observable (Measurement) Equations

For `order=2`, observables are currently restricted to be **affine in current-period endogenous variables**
(no lags/leads and no nonlinear transformations). This matches the common “linear measurement equation” setup and
keeps the particle filter well-defined.

If your model uses nonlinear observables (e.g. `xobs: exp(x)`), you’ll need to:

- change the observable definition (preferred), or
- compile/order=1 (linearized measurement), or
- compile/order=2 with `nonlinear_observables="linearize"` to interpret observables via a first-order linearization
  at the steady state (recommended only if you understand the approximation), or
- introduce explicit measurement-error structure and/or an auxiliary-state construction (future work).

## Determinacy and QZ Threshold

For linear (order-1) models, determinacy classification can depend on the QZ threshold near the unit circle.
You can pass `qz_criterium` through to the GENSYS solver:

```python
lin = m.compile_model(order=1)
TT, RR, RC = lin.solve_LRE(p0, qz_criterium=1 + 1e-6)
```

There is also a convenience report:

```python
rep = lin.determinacy_report(p0, qz_criteria=[1 + 1e-6, 1 + 1e-8])
print(rep)
```

## Dynare Cross-Checks (Optional)

This repo includes a YAML → Dynare `.mod` exporter:

```python
from dsge.dynare_export import to_dynare_mod

dyn = to_dynare_mod(m, order=2, pruning=True, irf=0)
print(dyn.mod_text[:200])
```

An opt-in integration test can run Dynare locally and compare 2nd-order decision-rule arrays:

```bash
DSGE_RUN_DYNARE=1 .venv/bin/python -m pytest -q dsge/tests/test_dynare_integration_second_order.py
```

If your `dynare` wrapper can’t locate Dynare’s MATLAB/Octave files, set `DYNARE_ROOT` to the Dynare source root
(the directory containing `matlab/dynare.m`).
