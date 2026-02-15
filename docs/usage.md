# Usage

## Load and compile a model
```python
from dsge import read_yaml
m = read_yaml('dsge/examples/fhp/fhp.yaml')  # or 'dsge/examples/edo/edo.yaml'
lin = m.compile_model()
CC, TT, RR, QQ, DD, ZZ, HH = lin.system_matrices(m.p0())
```

## Endogenous horizons (switching FHP)
If an FHP YAML includes `declarations.stopping_rule` (alias: `declarations.horizon_choice`), `read_yaml(...)` returns an `EndogenousHorizonSwitchingModel` (not an `FHPRepAgent`), so there is no `compile_model()` step:
```python
from dsge import read_yaml
m = read_yaml('dsge/examples/fhp/partial_equilibrium_endogenous.yaml')
sim = m.simulate(params=m.p0, T=200, seed=123)
girf = m.girf(m.p0, shock='e_y', h=20, reps=200, seed=123)
```

## Validate calibration and priors
```python
from dsge.validation import validate_model
validate_model(m)  # raises on structural issues
logp = lin.prior.logpdf(m.p0())
```

## Run tests (repo)
```bash
uv run -m pytest -q dsge/tests
# or
uv run python -m unittest discover -s dsge/tests -p 'test_*.py'
```

Tips:
- Keep paths relative (avoid absolute user paths in YAML).
- Use examples under `dsge/examples/` as templates for your models.
