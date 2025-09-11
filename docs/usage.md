# Usage

## Load and compile a model
```python
from dsge import read_yaml
m = read_yaml('dsge/examples/fhp/fhp.yaml')  # or 'dsge/examples/edo/edo.yaml'
lin = m.compile_model()
CC, TT, RR, QQ, DD, ZZ, HH = lin.system_matrices(m.p0())
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
