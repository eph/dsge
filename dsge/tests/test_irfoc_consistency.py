import io

import numpy as np
from numpy.testing import assert_allclose

from unittest import TestCase

from dsge.irfoc import IRFOC
from dsge.parse_yaml import read_yaml


_BASE_YAML = """
declarations:
  name: nk_irfoc_consistency
  variables: [pi, y, i, u, re, ilag]
  parameters: [beta, kappa, sigma, rho, gamma_pi, gamma_y, rho_u, rho_r]
  shocks: [eu, er, em]

equations:
  - pi = beta*pi(+1) + kappa*y + u
  - y = y(+1) - sigma*(i - pi(+1) - re)
  - i = rho*i(-1) + (1-rho)*(gamma_pi*pi + gamma_y*y) + em
  - u = rho_u*u(-1) + eu
  - re = rho_r*re(-1) + er
  - ilag = i(-1)

calibration:
  parameters:
    beta: 0.99
    kappa: 0.024
    sigma: 6.25
    rho: 0.85
    gamma_pi: 1.50
    gamma_y: 0.50
    rho_u: 0.6
    rho_r: 0.6
  covariance:
    eu: 1.0
    er: 1.0
    em: 1.0
"""


def _yaml_with_policy_params(rho: float, gamma_pi: float, gamma_y: float) -> str:
    # Keep the same parameter names; only the calibration changes.
    return _BASE_YAML.replace("rho: 0.85", f"rho: {rho}").replace("gamma_pi: 1.50", f"gamma_pi: {gamma_pi}").replace(
        "gamma_y: 0.50", f"gamma_y: {gamma_y}"
    )


class TestIRFOCConsistency(TestCase):
    def test_irfoc_matches_recompiled_model_with_changed_rule(self):
        """
        If we change the Taylor-rule coefficients in the model, that should match using IRFOC
        on the original model with `em` as the instrument shock enforcing the new rule.
        """
        m0 = read_yaml(io.StringIO(_yaml_with_policy_params(0.85, 1.50, 0.50)))
        m1 = read_yaml(io.StringIO(_yaml_with_policy_params(0.70, 2.00, 0.25)))

        lin0 = m0.compile_model()
        lin1 = m1.compile_model()
        p0 = m0.p0()
        p1 = m1.p0()

        T = 25
        shock_size = -2.0
        cols = ["pi", "y", "i", "ilag"]

        baseline0 = shock_size * lin0.impulse_response(p0, h=T - 1)["er"].loc[:, cols]
        target1 = shock_size * lin1.impulse_response(p1, h=T - 1)["er"].loc[:, cols]

        irfoc = IRFOC(m0, baseline0, instrument_shocks="em", p0=p0, compiled_model=lin0)
        rule = "i = 0.70*ilag + (1-0.70)*(2.0*pi + 0.25*y)"
        sim = irfoc.simulate(rule)

        # This is an identity in exact arithmetic; allow tiny numerical error from repeated solves/IRF construction.
        assert_allclose(sim[cols].to_numpy(), target1[cols].to_numpy(), rtol=0.0, atol=1e-5)

    def test_irfoc_matches_recompiled_model_for_another_shock(self):
        m0 = read_yaml(io.StringIO(_yaml_with_policy_params(0.85, 1.50, 0.50)))
        m1 = read_yaml(io.StringIO(_yaml_with_policy_params(0.70, 2.00, 0.25)))

        lin0 = m0.compile_model()
        lin1 = m1.compile_model()
        p0 = m0.p0()
        p1 = m1.p0()

        T = 20
        shock_size = 1.5
        cols = ["pi", "y", "i", "ilag"]

        baseline0 = shock_size * lin0.impulse_response(p0, h=T - 1)["eu"].loc[:, cols]
        target1 = shock_size * lin1.impulse_response(p1, h=T - 1)["eu"].loc[:, cols]

        irfoc = IRFOC(m0, baseline0, instrument_shocks="em", p0=p0, compiled_model=lin0)
        rule = "i = 0.70*ilag + (1-0.70)*(2.0*pi + 0.25*y)"
        sim = irfoc.simulate(rule)

        assert_allclose(sim[cols].to_numpy(), target1[cols].to_numpy(), rtol=0.0, atol=1e-5)
