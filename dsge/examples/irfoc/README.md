# IRFOC examples

This folder contains runnable examples for IRF-based optimal control / counterfactuals (`dsge.irfoc.IRFOC`).

## NK ZLB demo (policy-rule comparison)

Run:

```bash
.venv/bin/python dsge/examples/irfoc/nk_zlb_rules_demo.py
```

This simulates a simple NK model under three monetary policy rules in response to a large negative demand shock
(implemented as a negative natural-rate shock) and enforces a ZLB via `max(zlb, rule)` in IRFOC.

Notes:
- The model is linearized, so `i` is a *deviation* from steady state. The demo picks a toy steady-state nominal
  rate `i_ss` and uses `zlb = -i_ss` so that `i_level = i + i_ss` respects `i_level >= 0`.
- Output files (CSVs + PNG) are written to `dsge/examples/irfoc/_out/`.
