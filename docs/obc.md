# Occasionally Binding Constraints (OBC)

This guide introduces support for occasionally binding constraints, such as the zero lower bound (ZLB) on interest rates, using a regimes + constraints approach and an OccBin-style solver.

## YAML Structure

Add `regimes` and `constraints` alongside the usual `declarations` and `calibration`:

```yaml
declarations:
  name: nk_zlb
  variables: [y, pi, r]
  parameters: [beta, phi_pi, phi_y]
  shocks: [e]

calibration:
  parameters:
    beta: 0.99
    phi_pi: 1.5
    phi_y: 0.1
  covariance:
    e: 1.0

regimes:
  normal:
    equations:
      model:
        - r = phi_pi*pi + phi_y*y + e
        - y = y(-1) - r + e
        - pi = 0.99*pi(+1) + y
  zlb:
    equations:
      model:
        - r = 0
        - y = y(-1) - r + e
        - pi = 0.99*pi(+1) + y

constraints:
  - name: zlb
    when: "r < 0"
    binding_regime: zlb
    normal_regime: normal
    horizon: 40    # optional OccBin horizon
```

- `regimes`: Each regime defines its own `equations` (and optionally `observables`).
- `constraints`: A list of inequality triggers; currently a single-constraint workflow is supported.

## Solving (OccBin‑style)

The solver compiles a linear DSGE for each regime, then iterates over a candidate time path, switching regimes when the constraint condition `when` holds, until the regime sequence converges (no more flips). In this initial release, a placeholder IRF uses the normal regime.

## API (preview)

```python
from dsge import read_yaml
m = read_yaml('path/to/your_obc_model.yaml')  # returns OccBinModel
p0 = m.p0()
res = m.irf_obc(p0, h=20)
# res = { 'irf': {shock->DataFrame}, 'binding': array[(h+1), bool] }
```

## Notes

- Variable/shock sets must match across regimes.
- Observables are shared unless overridden per regime.
- Filtering/estimation under OBC is out of scope for the initial version.

Further iterations will add a full OccBin path solver, observable IRFs, and examples.

