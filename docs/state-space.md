# State-Space Models

This page describes the linear Gaussian state-space API and the DSGE specialization.

## Model Form

We work with the linear system parameterized by a vector `para`:

- State transition: `s_t = T(para) s_{t-1} + R(para) ε_t`, with `ε_t ~ N(0, Q(para))`
- Measurement: `y_t = D(para) + Z(para) s_t + η_t`, with `η_t ~ N(0, H(para))`

The base class `StateSpaceModel` expects callables `CC, TT, RR, QQ, DD, ZZ, HH` to build these matrices from `para`. Most users will use `LinearDSGEModel`, which solves a linear DSGE system (GAM0/GAM1/PSI/PPI) via `gensys` to produce `T, R`.

## Common Tasks

- Log-likelihood: `log_lik(para)`
- Filtering + smoothing: `kf_everything(para)`
- Impulse responses (states): `impulse_response(para, h=20)`
- ABCD representation: `abcd_representation(para)`
- Simulation: `simulate(para, nsim=200)` and `pred(para, ...)`
- Perfect foresight simulation: `perfect_foresight(para, eps_path, ...)`

## Minimal Example

```python
import numpy as np
from dsge.StateSpaceModel import StateSpaceModel

# AR(1) state with measurement y_t = s_t (no measurement error)
phi = 0.9
sigma = 1.0

CC = lambda p: np.array([0.0])               # (ns,)
TT = lambda p: np.array([[phi]])             # (ns x ns)
RR = lambda p: np.array([[1.0]])             # (ns x neps)
QQ = lambda p: np.array([[sigma**2]])        # (neps x neps)
DD = lambda p: np.array([0.0])               # (nobs,)
ZZ = lambda p: np.array([[1.0]])             # (nobs x ns)
HH = lambda p: np.array([[0.0]])             # (nobs x nobs)

# Dummy data (T x nobs)
yy = np.zeros((100, 1))
ssm = StateSpaceModel(yy, CC, TT, RR, QQ, DD, ZZ, HH,
                      state_names=['s'], obs_names=['y'], shock_names=['eps'])

para = []
print('log-lik:', ssm.log_lik(para))
print('IRFs:', ssm.impulse_response(para, h=10)['eps'].head())
```

## API Reference

- Base model:

::: dsge.StateSpaceModel.StateSpaceModel

- DSGE specialization:

::: dsge.StateSpaceModel.LinearDSGEModel

::: dsge.StateSpaceModel.LinearDSGEModelwithSV

