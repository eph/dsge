# Models

This repository supports three model families with distinct YAML layouts: Standard LRE (linear rational expectations), FHP representative‑agent, and SI (Sticky Information). All compile to a linear state‑space and expose the same matrices.

## Common Workflow
- Author a YAML model under `dsge/examples/<name>/`.
- Load/compile: `m = read_yaml(path); lin = m.compile_model(); p0 = m.p0()`.
- Get matrices: `CC, TT, RR, QQ, DD, ZZ, HH = lin.system_matrices(p0)`.

## Standard LRE (DSGE)
Use for canonical linearized NK/RBC models with forward‑looking equations.

YAML skeleton:
```yaml
declarations:
  name: nk
  variables: [y, c, pi, r]
  shocks: [g, mp]
  innovations: [eps_g, eps_mp]
  parameters: [beta, sigma, kappa, phi_pi, phi_y, rho_g, rho_mp]
model:
  static:                       # identities / definitions
    - y = c + g
    - r = phi_pi*pi + phi_y*y + mp
  cycle:                        # forward-looking block
    plan:
      - c = c(+1) - sigma*(r - pi(+1))        # Euler
      - pi = beta*pi(+1) + kappa*y            # NKPC
  shocks:                       # AR(1) shocks
    - g = rho_g*g(-1) + eps_g
    - mp = rho_mp*mp(-1) + eps_mp
  observables:                  # optional measurement mapping
    ygr: y - y(-1)
    pinf: 4*pi
calibration:
  parameters: { beta: 0.99, sigma: 1.0, kappa: 0.05, phi_pi: 1.5, phi_y: 0.125, rho_g: 0.9, rho_mp: 0.5 }
  covariance: { eps_g: 1.0, eps_mp: 1.0 }
estimation:
  prior: { sigma: [gamma, 1.0, 0.5], kappa: [gamma, 0.05, 0.05] }
```

Examples: `dsge/examples/edo/`, `dsge/examples/nkmp/`.

## FHP Representative‑Agent
Cycle/trend decomposition with a value block for latent value updates. See `dsge/FHPRepAgent.py` and `dsge/examples/fhp/`.

YAML skeleton (excerpt):
```yaml
declarations:
  type: fhp
  variables: [kp, c, pi, q, i, y, mc, r, y_lag]
  shocks: [re, mu, chi, g, mp]
  innovations: [e_re, e_mu, e_chi, e_g, e_mp]
  parameters: [sigma, sg, phi_k, delta, alpha, phi_pi, phi_y, kappa, r_A, rho_mu, gamma, rho_re, rho_chi, rho_g, rho_mp, y_Q, pi_A,
               sigma_mu, sigma_chi, sigma_g, sigma_mp, sigma_re]
  auxiliary_parameters: [beta, sc, r_k]
  # Finite-horizon depth for forward-looking recursion.
  # Either a scalar (applies to all forward-looking equation rows)...
  k: 1
  # ...or a row-specific spec keyed by equation LHS variable name:
  # k:
  #   default: 4
  #   by_lhs:
  #     pi: 1
model:
  static:
    - q = -mu + phi_k*(i - kp(-1))
    - i = -mu + 1/delta*(kp - (1-delta)*kp(-1))
    - y = sc*c + (1-sc-sg)*i + sg*g
    - mc = 1/sigma*c + chi + alpha/(1-alpha)*(y - kp(-1))
    - r = phi_pi*pi + phi_y*y + mp
  cycle:
    plan:
      - q = -(r - pi(+1) - re) + beta*q(+1) + beta*r_k*(mc(+1) + y(+1) - kp)
      - c = c(+1) - sigma*(r - pi(+1) - re)
      - pi = beta*pi(+1) + kappa*mc
  shocks:
    - re = rho_re*re(-1) + e_re
    - mu = rho_mu*mu(-1) + e_mu
    - chi = rho_chi*chi(-1) + e_chi
    - g = rho_g*g(-1) + e_g
    - mp = rho_mp*mp(-1) + e_mp
  trend:
    plan:
      - q = -(r - pi(+1)) + beta*q(+1) + beta*r_k*(mc(+1) + y(+1) - kp)
      - c = c(+1) - sigma*(r - pi(+1))
      - pi = beta*pi(+1) + kappa*mc
  value:
    update:
      - vke = beta*q + beta*delta*mu + beta/alpha*r_k*(mc - (1-alpha)*chi) - (1 + beta*r_k*(1-alpha)/alpha)/sigma*c
      - vhe = c + sigma*pi
      - vfe = pi
  observables:
    ygr: y_Q + y - y_lag
    pinf: pi_A + 4*pi
    nomr: r_A + pi_A + 4*r
calibration:
  parameters: { sigma: 1, sg: 0.15, phi_k: 1.0, delta: 0.025, alpha: 0.30, phi_pi: 1.5, phi_y: 0.125, kappa: 0.01, gamma: 0.1, r_A: 2 }
  covariance: { e_re: 1, e_mu: 1, e_chi: 1, e_g: 1, e_mp: 1 }
```

### Mixed horizons (row-specific `k`)
If `declarations.k` is a dict with `default` and `by_lhs`, the recursion uses a per-equation-row horizon `k_i`:
- Each `by_lhs` key must match the LHS variable name in your YAML equation (e.g. `pi = ...`).
- Internally, the model still stores a scalar `k = max_i k_i` for backward compatibility, plus `k_spec`.

Example: firms price with short horizon, households plan longer:
```yaml
declarations:
  k:
    default: 4
    by_lhs:
      pi: 1
```

See `dsge/examples/fhp/fhp_mixed_k.yaml`.

### Endogenous horizons (state-dependent switching)
You can optionally add an endogenous stopping rule that selects a discrete horizon each period for one or more components.

In FHP YAML this lives under `declarations.stopping_rule` (alias: `declarations.horizon_choice`) and defines:
- components (e.g. `pricing`, `hh`), each with a `k_max` and a list of equation rows (`assign_lhs`)
- a constant marginal cost `a` and a curvature/normalization `lambda`
- a `policy_object` (an expression evaluated on model observables + a reduced switching state)

Example (partial equilibrium pricing block):
```yaml
declarations:
  stopping_rule:
    components:
      pricing:
        k_max: 8
        assign_lhs: [pi]
        cost: { a: 1e-4 }
        lambda: "(-D_pp)/(1-beta*theta)"
        policy_object: "theta/(1-theta) * pi"
```

`policy_object` may reference any declared parameter plus observable names. If you declare no observables, FHP defaults to using the model variables as observables (identity mapping).

Important: when `stopping_rule`/`horizon_choice` is present, `read_yaml(...)` returns an `EndogenousHorizonSwitchingModel` (piecewise-linear, regime-dependent matrices), not a fixed-regime `FHPRepAgent`. Use simulation / particle filtering rather than standard linear IRFs.

See `dsge/examples/fhp/partial_equilibrium_endogenous.yaml` and `dsge/tests/test_fhp_endogenous_horizon_pe_yaml.py`.

Notes: Validators forbid future‑dated shocks and check dimensions/usage; see `dsge/tests/test_fhp.py` and `dsge/tests/test_fhp_validation.py`.

## SI (Sticky Information)
Compact “sticky‑information” style models: equations at the top level; optional index for aggregation. See `dsge/SIDSGE.py` and `dsge/examples/si/`.

YAML skeleton (based on Mankiw–Reis):
```yaml
declarations:
  name: mr
  type: sticky-information
  index: j
  variables: [pp, y, ygr, delm]
  shocks: [e]
  parameters: [alp, lam, sigm, rho]
equations:
  - ygr + pp = -sigm*delm
  - alp*lam/(1-lam)*y + lam * SUM((1-lam)^j * EXP(-j-1) (pp + alp*ygr), (j, 0, inf)) = pp
  - ygr = y - y(-1)
  - delm = rho*delm(-1) + e
calibration:
  parameters: { alp: 0.1, lam: 0.25, rho: 0.5, sigm: 0.007 }
```

Examples: `dsge/examples/si/mankiw-reis.yaml`.

## Validation & Common Checks
- Symbol hygiene: all variables/shocks/parameters used are declared; no typos.
- Temporal logic: no future‑dated shocks in equations; leads/lags are integers.
- Observables: mappings are valid and dimensionally consistent; data (if provided) aligns with observables.
- Priors: `lin.prior.logpdf(p0)` evaluates; covariance matches innovations.
- Run tests: `uv run -m pytest -q dsge/tests` (see validation‑focused tests for coverage).

## State‑Space Output
```python
from dsge import read_yaml
m = read_yaml('dsge/examples/nkmp/nkmp.yaml')
lin = m.compile_model(); p0 = m.p0()
CC, TT, RR, QQ, DD, ZZ, HH = lin.system_matrices(p0)
```
- `TT, RR, QQ`: transition and shock structure; `DD, ZZ, HH`: measurement.

## References
- Sims, C. A. (2002). Solving Linear Rational Expectations Models. Computational Economics, 20(1), 1–20. [GENSYS]
- Woodford, M. (2003). Interest and Prices: Foundations of a Theory of Monetary Policy. Princeton University Press.
- Mankiw, N. G., & Reis, R. (2002). Sticky Information versus Sticky Prices. Quarterly Journal of Economics, 117(4), 1295–1328.
