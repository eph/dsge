# Quickstart

This guide shows a minimal end-to-end flow: install, load a model, compile it, and get system matrices/IRFs.

Prerequisites
- Python 3.9+
- Optional: uv for fast, reproducible runs (recommended)

Install (dev)
- Editable install with uv:
  - `uv pip install -e .`
- Or with pip:
  - `pip install -e .`

Load a simple model
```python
from dsge import read_yaml

# Use a tiny AR(1) example bundled with the repo
m = read_yaml('dsge/examples/ar1/ar1.yaml')
print(m)
```

Compile and get system matrices
```python
# Linearize and construct the state-space representation
lin = m.compile_model()

# Calibrated parameter vector (ordering follows m.parameters)
p0 = m.p0()

# System matrices
CC, TT, RR, QQ, DD, ZZ, HH = lin.system_matrices(p0)
print('TT shape:', TT.shape)
```

Impulse responses (IRFs)
```python
# A quick IRF of length 20 periods
irfs = lin.impulse_response(p0, h=20)
# irfs is a dict keyed by shock name -> pandas DataFrame of responses
shock = list(irfs.keys())[0]
print('Shocks:', list(irfs.keys()))
print(irfs[shock].head())
```

Working with examples
- Explore more YAMLs in `dsge/examples/` (e.g., `nkmp`, `sw`, `DGS`, `fhp`).
- Validate and inspect equations via `print(m)` or `m.equations`.

Docs and DSL
- Parsing syntax for equations: see `Parsing DSL`.

Notebook
- Downloadable: [Quickstart Notebook](notebooks/quickstart.ipynb)

Run tests (optional)
- `uv run -m pytest -q dsge/tests`
```
# Quick import check
uv run python -c "import dsge; print(dsge.__file__)"
```
