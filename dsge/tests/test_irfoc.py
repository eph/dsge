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


def _simple_nk_yaml_with_extra_lag_aliases() -> str:
    return """
declarations:
  name: irfoc_smoke_extra_lags
  variables: [pi, y, i, u, re, ilag, pilag, ylag]
  parameters: [beta, kappa, sigma, rho, gamma_pi, gamma_y, rho_u, rho_r]
  shocks: [eu, er, em]

equations:
  - pi = beta*pi(+1) + kappa*y + u
  - y = y(+1) - sigma*(i - pi(+1) - re)
  - i = rho*i(-1) + (1-rho)*(gamma_pi*pi + gamma_y*y) + em
  - u = rho_u*u(-1) + eu
  - re = rho_r*re(-1) + er
  - ilag = i(-1)
  - pilag = pi(-1)
  - ylag = y(-1)

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


def test_irfoc_affine_rule_with_lagged_variable():
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml()))
    lin = m.compile_model()
    p0 = m.p0()

    T = 40
    cols = ["pi", "y", "i"]
    baseline = lin.impulse_response(p0, h=T - 1)["er"].loc[:, cols]

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)
    sim = irfoc.simulate("i = 1.7*pi + 0.2*y + 0.9*i(-1)", return_details=False)

    i = sim["i"].to_numpy()
    pi = sim["pi"].to_numpy()
    y = sim["y"].to_numpy()
    i_lag = np.r_[0.0, i[:-1]]
    resid = i - (1.7 * pi + 0.2 * y + 0.9 * i_lag)
    assert float(np.max(np.abs(resid))) < 1e-7


@pytest.mark.parametrize(
    "rule",
    [
        "i = 1.2*pi(-1) + 0.3*y",
        "i = 0.7*y(-1) + 0.2*pi + 0.5*i(-1)",
        "i = 0.4*pi(-2) + 0.3*y",
    ],
)
def test_irfoc_affine_rule_with_lagged_other_endogenous_variables(rule):
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml()))
    lin = m.compile_model()
    p0 = m.p0()

    T = 20
    baseline = lin.impulse_response(p0, h=T - 1)["er"].loc[:, ["pi", "y", "i", "ilag"]]

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)
    res = irfoc.simulate(rule, return_details=True)

    assert res.residuals.shape == (T, 1)
    assert float(np.max(np.abs(res.residuals.to_numpy()))) < 1e-9


def test_irfoc_lagged_rule_returns_residuals():
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml()))
    lin = m.compile_model()
    p0 = m.p0()

    T = 20
    baseline = lin.impulse_response(p0, h=T - 1)["er"].loc[:, ["pi", "y", "i"]]

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)
    res = irfoc.simulate("i = 0.5*i(-1)", return_details=True)

    assert res.residuals.shape == (T, 1)
    assert float(np.max(np.abs(res.residuals.to_numpy()))) < 1e-9


def test_irfoc_explicit_lag_matches_alias_on_shifted_baseline():
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml()))
    lin = m.compile_model()
    p0 = m.p0()

    full = lin.impulse_response(p0, h=12)["er"].loc[:, ["pi", "y", "i", "ilag"]]
    baseline = full.iloc[1:].copy()
    baseline.index = range(len(baseline))
    baseline["ilag"] = full["i"].iloc[:-1].to_numpy()

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)
    explicit = irfoc.simulate("i = 0.5*i(-1)")
    alias = irfoc.simulate("i = 0.5*ilag")

    np.testing.assert_allclose(
        explicit[["pi", "y", "i", "ilag"]].to_numpy(),
        alias[["pi", "y", "i", "ilag"]].to_numpy(),
        rtol=0.0,
        atol=1e-9,
    )


def test_irfoc_explicit_other_lags_match_aliases_on_shifted_baseline():
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml_with_extra_lag_aliases()))
    lin = m.compile_model()
    p0 = m.p0()

    full = lin.impulse_response(p0, h=12)["er"].loc[:, ["pi", "y", "i", "ilag", "pilag", "ylag"]]
    baseline = full.iloc[1:].copy()
    baseline.index = range(len(baseline))
    baseline["ilag"] = full["i"].iloc[:-1].to_numpy()
    baseline["pilag"] = full["pi"].iloc[:-1].to_numpy()
    baseline["ylag"] = full["y"].iloc[:-1].to_numpy()

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)
    explicit = irfoc.simulate("i = 0.5*pi(-1) + 0.2*y(-1) + 0.1*i(-1)")
    alias = irfoc.simulate("i = 0.5*pilag + 0.2*ylag + 0.1*ilag")

    np.testing.assert_allclose(
        explicit[["pi", "y", "i", "ilag", "pilag", "ylag"]].to_numpy(),
        alias[["pi", "y", "i", "ilag", "pilag", "ylag"]].to_numpy(),
        rtol=0.0,
        atol=1e-9,
    )


def test_irfoc_piecewise_explicit_lag_matches_alias_on_shifted_baseline():
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml()))
    lin = m.compile_model()
    p0 = m.p0()

    full = lin.impulse_response(p0, h=12)["er"].loc[:, ["pi", "y", "i", "ilag"]]
    baseline = full.iloc[1:].copy()
    baseline.index = range(len(baseline))
    baseline["ilag"] = full["i"].iloc[:-1].to_numpy()

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)
    explicit = irfoc.simulate("i = max(0.001, 0.5*i(-1))", return_details=True)
    alias = irfoc.simulate("i = max(0.001, 0.5*ilag)", return_details=True)

    np.testing.assert_allclose(
        explicit.simulation[["pi", "y", "i", "ilag"]].to_numpy(),
        alias.simulation[["pi", "y", "i", "ilag"]].to_numpy(),
        rtol=0.0,
        atol=1e-9,
    )
    assert explicit.residuals.shape == (len(baseline), 1)
    assert float(np.max(np.abs(explicit.residuals.to_numpy()))) < 1e-9


def test_irfoc_max_min_not_implemented():
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml()))
    lin = m.compile_model()
    p0 = m.p0()
    baseline = lin.impulse_response(p0, h=10)["er"].loc[:, ["pi", "y", "i", "ilag"]]

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)

    sim = irfoc.simulate("i = max(0.002, 1.5*pi + 0.1*y + 0.9*ilag)", return_details=False)
    rhs = np.maximum(
        0.002,
        (1.5 * sim["pi"] + 0.1 * sim["y"] + 0.9 * sim["ilag"]).to_numpy(),
    )
    assert np.max(np.abs(sim["i"].to_numpy() - rhs)) < 1e-7


def test_irfoc_indicator_syntax_one_parens():
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml()))
    lin = m.compile_model()
    p0 = m.p0()
    baseline = lin.impulse_response(p0, h=10)["er"].loc[:, ["pi", "y", "i", "ilag"]]

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)

    sim = irfoc.simulate("i = 0.002 + 0.001*1(pi < 0)", return_details=False)
    vals = sim["i"].to_numpy()
    # Should be close to one of the two regime levels each period.
    dist = np.minimum(np.abs(vals - 0.002), np.abs(vals - 0.003))
    assert float(np.max(dist)) < 1e-6


def test_irfoc_indicator_rejects_bilinear():
    import io

    m = read_yaml(io.StringIO(_simple_nk_yaml()))
    lin = m.compile_model()
    p0 = m.p0()
    baseline = lin.impulse_response(p0, h=10)["er"].loc[:, ["pi", "y", "i", "ilag"]]

    irfoc = IRFOC(m, baseline, instrument_shocks="em", p0=p0, compiled_model=lin)

    with pytest.raises(ValueError, match="scalar \\* affine"):
        irfoc.simulate("i = 1(pi < 0)*pi")
