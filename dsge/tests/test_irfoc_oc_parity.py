import io

import numpy as np

from numpy.testing import assert_allclose

from dsge.irfoc import IRFOC
from dsge.oc import compile_commitment
from dsge.parse_yaml import read_yaml


def _simple_nk_with_deli_yaml() -> str:
    return """
declarations:
  name: 'example_dsge'
  variables: [pi, y, i, u, re]
  parameters: [beta, kappa, sigma, gamma_pi, gamma_y, rho_u, rho_r]
  shocks: [eu, er, em]

equations:
  - pi = beta * pi(+1) + kappa * y + u
  - y = y(+1) - sigma * (i - pi(+1) - re)
  - i = gamma_pi * pi + gamma_y * y + em
  - u = rho_u * u(-1) + eu
  - re = rho_r * re(-1) + er

calibration:
  parameters:
    beta: 0.99
    kappa: 0.024
    sigma: 6.25
    gamma_pi: 1.50
    gamma_y: 0.15
    rho_u: 0.0
    rho_r: 0.50
"""


def test_irfoc_optimal_control_matches_oc_commitment_irf():
    """
    For the simple NK model with a policy shock entering additively, choosing the
    sequence of policy shocks to minimize a quadratic loss should replicate the
    deterministic commitment OC impulse response (up to numerical tolerance).
    """
    m = read_yaml(io.StringIO(_simple_nk_with_deli_yaml()))
    lin = m.compile_model()
    p0 = m.p0()

    loss = "pi**2 + y**2"
    T = 40
    cols = ["pi", "y", "i", "u", "re"]

    baseline = lin.impulse_response(p0, h=T - 1)["er"].loc[:, cols]
    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)
    sim = irfoc.simulate_optimal_control(loss, discount="beta", ridge=1e-12, return_details=False)

    oc_mod = compile_commitment(m, loss, "i", "em", beta="beta")
    oc_irf = oc_mod.impulse_response(p0, h=T - 1)["er"].loc[:, ["pi", "y", "i"]]

    assert_allclose(sim.loc[:, ["pi", "y", "i"]].to_numpy(), oc_irf.to_numpy(), atol=3e-6, rtol=1e-6)
