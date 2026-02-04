dsge
====
![Coverage](badges/coverage.svg)
A simple Python (3+) package for Dynamic Stochastic General Equilibrium (DSGE) models.

This was originally forked from Pablo Winant's (excellent) package
dolo.  (See [https://github.com/EconForge/dolo].)  I wrote this mainly
for my own personal use; as such it may contain bugs, and the
documentation might be lacking in some (many) places.

Installation
------------

- From PyPI (if published):
  - `pip install dsge`
  - `uv pip install dsge`
  - `poetry add dsge`

- From source (this repository):
  - Using uv (recommended for dev): `uv pip install -e .`
  - Using pip: `pip install -e .`
  - Using Poetry (no Poetry metadata required):
    - `poetry shell` then `pip install -e '.[test]'` (installs package + test extras), or
    - `poetry run pip install -e '.[test]'`

Tests
-----

- Using uv (pytest): `uv run -m pytest -q dsge/tests`
- Using uv (unittest): `uv run python -m unittest discover -s dsge/tests -p 'test_*\.py'`
- Using Poetry (ensure test deps are installed first, e.g., `poetry run pip install -e '.[test]'`):
  - Pytest: `poetry run pytest -q dsge/tests`
  - Unittest: `poetry run python -m unittest discover -s dsge/tests -p 'test_*\.py'`

Quick Check
-----------

- Import check: `uv run python -c "import dsge; print(dsge.__file__)"`

Second-Order Perturbation (Experimental)
---------------------------------------

For standard LRE DSGE models, you can compute a second-order perturbation solution:

- Example:
  - `uv run python -c "from dsge.parse_yaml import read_yaml; m = read_yaml('dsge/examples/nkmp/nkmp.yaml'); sol = m.solve_second_order(m.p0()); print(sol.gx.shape, sol.gxx.shape)"`

`solve_second_order()` returns a `SecondOrderSolution` with first- and second-order decision-rule coefficients
and a small `as_dynare_like()` export helper for comparing arrays against Dynare outputs after reordering.

You can also compile an order-2 model with the same high-level API as the linear model:

- `c = m.compile_model(order=2)` (uses pruned order-2 decision rules + particle filter likelihood)
- `c.log_lik(p0, nparticles=2000, seed=0)` (PF tuning lives on the likelihood call)

Dynare Cross-Checks (Optional)
------------------------------

This repo includes a basic YAML→Dynare `.mod` exporter (`dsge.dynare_export.to_dynare_mod`) so you can run Dynare
locally and compare results. Dynare itself is not bundled.

There is also an (opt-in) integration test that runs Dynare and compares second-order decision-rule arrays:

- `DSGE_RUN_DYNARE=1 uv run -m pytest -q dsge/tests/test_dynare_integration_second_order.py`

If your `dynare` wrapper can’t locate Dynare’s MATLAB/Octave files, set `DYNARE_ROOT` to the Dynare source root
(the directory containing `matlab/dynare.m`).

Bugs and Questions
------------------
For bug reports and questions, send me an email at
ed.herbst@gmail.com.  If there is enough interest,
I'll mirror the source on github.

Documentation
--------------

Build and serve the MkDocs site locally using uv:

- Install docs tooling (one-time):
  - `uv pip install mkdocs mkdocs-material mkdocs-macros-plugin "mkdocstrings[python]"`
  - or `make docs-install`

- Serve with live reload:
  - `uv run mkdocs serve`
  - or `make docs-serve`

- Build static site (outputs to `site/`):
  - `uv run mkdocs build --strict`
  - or `make docs-build`

Docs entry points:
- Overview: `docs/index.md`
- Parsing DSL: `docs/parsing.md`
- Config: `mkdocs.yml`
