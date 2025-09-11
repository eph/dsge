# DSGE Documentation

Current version: {{ version }}

Modern tools for building, validating, and working with Dynamic Stochastic General Equilibrium (DSGE) models in Python.

- YAML models → parsed and validated
- Linearization → state‑space matrices for estimation and simulation
- Utilities → priors, filtering, Markov switching, and more

Quick example:
```python
from dsge import read_yaml
m = read_yaml('dsge/examples/edo/edo.yaml')
lin = m.compile_model()
CC, TT, RR, QQ, DD, ZZ, HH = lin.system_matrices(m.p0())
```

Use the left navigation to explore guides and the API reference.
