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

## API

```python
from dsge import read_yaml
m = read_yaml('path/to/your_obc_model.yaml')  # returns OccBinModel
compiled = m.compile_model()
p0 = m.p0()

# Standardized IRF interfaces
states_irfs = compiled.impulse_response_states(p0, h=20)        # dict shock->DataFrame
obs_irfs    = compiled.impulse_response_observables(p0, h=20)   # dict shock->DataFrame

# Full OBC metadata (binding flags and regimes) for a specific shock
res_full = compiled.impulse_response(p0, h=20, shock_name='e_d')
# res_full has: 'states', 'observables', 'binding', 'regimes', and DataFrames if pandas is installed
```

Example files:
- YAML: `dsge/examples/nk_zlb/nk_zlb.yaml`
- Notebook: `docs/notebooks/nk_zlb.ipynb`

## Notes

- Variable/shock sets must match across regimes.
- Observables are shared unless overridden per regime.
- Filtering/estimation under OBC is out of scope for the initial version.
- Multiple constraints are supported if all map to the same (normal, binding) regime pair. Otherwise, a NotImplemented error is raised.

Further iterations will add a full OccBin path solver, observable IRFs, and examples.
