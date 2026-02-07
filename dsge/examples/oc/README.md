# Optimal control (OC) examples

This folder contains runnable examples for the Dennis-style optimal-control solvers in `dsge.oc`.

## NK commitment demo

Run:

```bash
.venv/bin/python dsge/examples/oc/nk_oc_demo.py
```

This script:
- compiles a baseline (Taylor rule) model,
- compiles a commitment optimal policy using `compile_commitment(...)`,
- (optionally) cross-checks against the IRFOC quadratic backend on the same loss,
- writes outputs to `dsge/examples/oc/_out/`.

