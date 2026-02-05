# Changelog

This project uses `setuptools_scm` (version comes from git tags).

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
