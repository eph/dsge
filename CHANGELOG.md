# Changelog

This project uses `setuptools_scm` (version comes from git tags).

## 0.3.0 (Unreleased)

- Big-model YAML load speed: use `xreplace` for lag/lead symbol rewrites (avoid expensive SymPy `subs`)
- Big-model compile speed: `python_sims_matrices(method="auto")` now avoids the full Jacobian backend on large systems
- Parsing: `find_max_lead_lag` rewritten to single-pass over equations (avoid O(n_symbols * n_atoms))
- LRE robustness: `solve_LRE(..., scale_equations=True)` row-scales equations to improve conditioning
- LRE diagnostics: `solve_LRE(..., realsmall=...)` and `determinacy_report(..., realsmall_criteria=...)` for sensitivity checks
- Gensys: improve “coincident zeros” detection and SVD tolerances using relative scaling diagnostics

## 0.2.7 (2026-02-07)

- IRFOC: add quadratic optimal-control (`simulate_optimal_control`)
- IRFOC: support indicator syntax `1(cond)` in piecewise rules
- OC/IRFOC: handle lagged policy instruments consistently
- OC: fix commitment solution when loss penalizes instruments (`Q != 0`)
- Tests: parity checks between OC and IRFOC for simple quadratic losses
- Examples: IRFOC NK ZLB rule comparison + OC NK commitment demo

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
