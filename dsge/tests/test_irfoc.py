import numpy as np
import pandas as pd

import pytest

from dsge.irfoc import IRFOC
from dsge.parse_yaml import read_yaml


def _simple_nk_yaml() -> str:
    return """
declarations:
  name: irfoc_smoke
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
    gamma_y: 0.15
    rho_u: 0.6
    rho_r: 0.6
"""


def test_irfoc_enforces_affine_rule():
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml()))
    lin = m.compile_model()
    p0 = m.p0()

    T = 40
    cols = ["pi", "y", "i", "ilag"]
    baseline = lin.impulse_response(p0, h=T - 1)["er"].loc[:, cols]
    assert isinstance(baseline, pd.DataFrame)

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)
    res = irfoc.simulate("i = 1.7*pi + 0.2*y + 0.9*ilag", return_details=True)

    # Rule residual should be essentially zero (numerical tolerance).
    max_abs = float(np.max(np.abs(res.residuals.values)))
    assert max_abs < 1e-9

    # Should move the baseline (unless it's already consistent with the rule).
    assert float(np.max(np.abs((res.simulation - baseline).values))) > 1e-10

    # Shocks path length and name.
    assert res.shocks.shape == (T, 1)
    assert list(res.shocks.columns) == ["em"]


def test_irfoc_rejects_nonlinear_rules():
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml()))
    lin = m.compile_model()
    p0 = m.p0()
    baseline = lin.impulse_response(p0, h=10)["er"].loc[:, ["pi", "y", "i", "ilag"]]

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)

    with pytest.raises(ValueError, match="affine"):
        irfoc.simulate("i = pi**2 + y")


def test_irfoc_max_min_not_implemented():
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml()))
    lin = m.compile_model()
    p0 = m.p0()
    baseline = lin.impulse_response(p0, h=10)["er"].loc[:, ["pi", "y", "i", "ilag"]]

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)

    with pytest.raises(NotImplementedError, match="max/min"):
        irfoc.simulate("i = max(0, 1.5*pi)")
