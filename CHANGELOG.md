# Changelog

This project uses `setuptools_scm` (version comes from git tags).

## 0.2.5 (2026-02-05)

- Add `dsge.irfoc.IRFOC` (IRF-based counterfactual simulator) with an affine policy-rule parser

## 0.2.6 (2026-02-05)

- IRFOC: support `max()`/`min()` (nested) piecewise-affine rules via MILP (`simulate_piecewise`)

## 0.2.2 (2026-02-05)

- Docs: `order=2` guide mentions `nonlinear_observables="linearize"` option
- CI: improve docs build caching key for MkDocs dependencies

## 0.2.1 (2026-02-05)

- CI: align test matrix with Python >= 3.10 and install MkDocs plugin dependencies
- Tests: add regression guard for `python_sims_matrices` cache early-return
- Docs: clarify `compile_model(..., nonlinear_observables=...)` and order-2 `log_lik` arguments

## 0.2.0 (2026-02-05)

- Second-order perturbation for LRE models: `DSGE.solve_second_order()` and `DSGE.compile_model(order=2)`
- Particle-filter likelihood with `nparticles` / `seed` options on `log_lik`
- YAML → Dynare `.mod` export + optional Dynare integration parity tests
- `SecondOrderSolution.as_dynare_like()`: align `ghxu`/`ghuu` flattening with Dynare conventions
- Order-2 compile: optional `nonlinear_observables="linearize"` to allow nonlinear observables via linearization
- Linear compilation: faster `python_sims_matrices(method="jacobian")` with safe fallback

## 0.1.1 (2026-02-05)

- `oc.py`: fix loss-matrix parsing for multiple policy instruments
- `dsge/tests/test_oc.py`: remove interactive plotting (`plt.show`) so tests are headless-friendly
- `LinearDSGEModel`: add `determinacy_report()` convenience wrapper + make `solve_LRE(use_cache=True)` safe when cache is empty

## 0.1.0 (2026-02-05)

- Gensys determinacy robustness improvements (`qz_criterium` support + diagnostics)
- Compilation speedups via caching for repeated matrix construction
