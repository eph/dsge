import io

import numpy as np

import pytest

from dsge.parse_yaml import read_yaml


def test_compile_model_order2_particle_filter_smoke():
    yaml_text = """
declarations:
  name: order2_smoke
  variables: [x, y]
  shocks: [e]
  parameters: [rho, beta, phi]
  observables: [x, y]
  measurement_errors: [me_x, me_y]

equations:
  model:
    - x = rho*x(-1) + e
    - y = beta*y(1) + x + phi/2*x^2
  observables:
    x: x
    y: y

calibration:
  parameters:
    rho: 0.9
    beta: 0.99
    phi: 0.5
  covariance:
    e: 0.01
  measurement_errors:
    me_x: 1e-4
    me_y: 1e-4
"""
    m = read_yaml(io.StringIO(yaml_text))
    p0 = m.p0()
    c = m.compile_model(order=2)

    y = np.zeros((20, 2))
    ll = c.log_lik(p0, y=y, nparticles=300, seed=0)
    assert np.isfinite(ll)

    irfs = c.impulse_response(p0, h=5)
    assert "e" in irfs
    assert irfs["e"].shape[0] == 6


def test_compile_model_order2_rejects_nonlinear_observables():
    yaml_text = """
declarations:
  name: order2_bad_obs
  variables: [x]
  shocks: [e]
  parameters: [rho]
  observables: [xobs]
  measurement_errors: [me_xobs]

equations:
  model:
    - x = rho*x(-1) + e
  observables:
    xobs: exp(x)

calibration:
  parameters:
    rho: 0.9
  covariance:
    e: 0.01
  measurement_errors:
    me_xobs: 1e-4
    """
    m = read_yaml(io.StringIO(yaml_text))
    with pytest.raises(ValueError, match="observable equations must be affine"):
        m.compile_model(order=2)


def test_compile_model_order2_allows_linearized_nonlinear_observables():
    yaml_text = """
declarations:
  name: order2_bad_obs_linearize
  variables: [x]
  shocks: [e]
  parameters: [rho]
  observables: [xobs]
  measurement_errors: [me_xobs]

equations:
  model:
    - x = rho*x(-1) + e
  observables:
    xobs: exp(x)

calibration:
  parameters:
    rho: 0.9
  covariance:
    e: 0.01
  measurement_errors:
    me_xobs: 1e-4
    """
    m = read_yaml(io.StringIO(yaml_text))
    p0 = m.p0()
    c = m.compile_model(order=2, nonlinear_observables="linearize")

    y = np.zeros((10, 1))
    ll = c.log_lik(p0, y=y, nparticles=250, seed=0)
    assert np.isfinite(ll)


def test_compile_model_order2_particle_filter_weight_propagation():
    """
    Regression: when not resampling, particle weights must be propagated across time.
    """
    yaml_text = """
declarations:
  name: order2_pf_weights
  variables: [x, y]
  shocks: [e]
  parameters: [rho, beta, phi]
  observables: [x, y]
  measurement_errors: [me_x, me_y]

equations:
  model:
    - x = rho*x(-1) + e
    - y = beta*y(1) + x + phi/2*x^2
  observables:
    x: x
    y: y

calibration:
  parameters:
    rho: 0.9
    beta: 0.99
    phi: 0.5
  covariance:
    e: 0.01
  measurement_errors:
    me_x: 1e-4
    me_y: 1e-4
"""
    m = read_yaml(io.StringIO(yaml_text))
    p0 = m.p0()
    c = m.compile_model(order=2)

    t = 25
    y = np.c_[np.linspace(-0.1, 0.2, t), np.linspace(0.05, -0.15, t)]

    nparticles = 250
    seed = 123
    ll = c.log_lik(p0, y=y, nparticles=nparticles, seed=seed, resample_threshold=1e-12)

    # Manual computation with persistent (normalized) log-weights.
    rng = np.random.default_rng(seed)
    pol = c._policy(p0, use_cache=False)
    dd, zz, hh, qq = c._measurement_matrices(p0, use_cache=False)

    nstate = pol.hx.shape[0]
    nshocks = qq.shape[0]
    cholqq = np.linalg.cholesky(qq)

    x1 = np.zeros((nparticles, nstate))
    x2 = np.zeros((nparticles, nstate))
    logw = np.full(nparticles, -np.log(nparticles), dtype=float)

    def logsumexp_1d(logx: np.ndarray) -> float:
        m = float(np.max(logx))
        return m + float(np.log(np.sum(np.exp(logx - m))))

    ll_manual = 0.0
    ll_buggy = 0.0

    for tt in range(t):
        y_t = np.asarray(y[tt]).reshape(-1)
        eps = rng.standard_normal(size=(nparticles, nshocks)) @ cholqq.T

        y1 = x1 @ pol.gx.T + eps @ pol.gu.T
        y2 = pol.gss + x2 @ pol.gx.T
        if c.pruning:
            y2 = (
                y2
                + 0.5 * np.einsum("kij,ni,nj->nk", pol.gxx, x1, x1)
                + np.einsum("kij,ni,nj->nk", pol.gxu, x1, eps)
                + 0.5 * np.einsum("kij,ni,nj->nk", pol.guu, eps, eps)
            )
        y_curr = y1 + y2
        yhat = dd + y_curr @ zz.T

        x1_next = x1 @ pol.hx.T + eps @ pol.hu.T
        x2_next = pol.hss + x2 @ pol.hx.T
        if c.pruning:
            x2_next = (
                x2_next
                + 0.5 * np.einsum("kij,ni,nj->nk", pol.hxx, x1, x1)
                + np.einsum("kij,ni,nj->nk", pol.hxu, x1, eps)
                + 0.5 * np.einsum("kij,ni,nj->nk", pol.huu, eps, eps)
            )

        x1, x2 = x1_next, x2_next

        resid = y_t[None, :] - yhat
        invhh = np.linalg.inv(hh)
        logdet = np.linalg.slogdet(hh)[1]
        quad = np.einsum("ni,ij,nj->n", resid, invhh, resid)
        logp = -0.5 * (quad + logdet + resid.shape[1] * np.log(2.0 * np.pi))

        logw = logw + logp
        inc = logsumexp_1d(logw)
        ll_manual += inc
        logw = logw - inc

        inc_buggy = logsumexp_1d(logp) - np.log(nparticles)
        ll_buggy += inc_buggy

    assert np.isfinite(ll)
    assert np.isclose(ll, ll_manual, rtol=0.0, atol=1e-9)
    assert abs(ll - ll_buggy) > 1e-6


def test_compile_model_order2_particle_filter_missing_data_correlated_meas_error():
    yaml_text = """
declarations:
  name: order2_pf_missing
  variables: [x, y]
  shocks: [e]
  parameters: [rho, beta, phi]
  observables: [x, y]
  measurement_errors: [me_x, me_y]

equations:
  model:
    - x = rho*x(-1) + e
    - y = beta*y(1) + x + phi/2*x^2
  observables:
    x: x
    y: y

calibration:
  parameters:
    rho: 0.9
    beta: 0.99
    phi: 0.5
  covariance:
    e: 0.01
  measurement_errors:
    me_x: 1e-4
    me_y: 2e-4
    me_x, me_y: 5e-5
"""
    m = read_yaml(io.StringIO(yaml_text))
    p0 = m.p0()
    c = m.compile_model(order=2)

    y = np.zeros((15, 2))
    y[::2, 1] = np.nan
    y[1::2, 0] = np.nan

    ll = c.log_lik(p0, y=y, nparticles=400, seed=0, resample_threshold=0.4)
    assert np.isfinite(ll)


def test_compile_model_order2_particle_filter_reproducible_seed():
    yaml_text = """
declarations:
  name: order2_pf_seed
  variables: [x]
  shocks: [e]
  parameters: [rho]
  observables: [x]
  measurement_errors: [me_x]

equations:
  model:
    - x = rho*x(-1) + e
  observables:
    x: x

calibration:
  parameters:
    rho: 0.9
  covariance:
    e: 0.01
  measurement_errors:
    me_x: 1e-4
"""
    m = read_yaml(io.StringIO(yaml_text))
    p0 = m.p0()
    c = m.compile_model(order=2)

    y = np.linspace(-0.05, 0.05, 20).reshape(-1, 1)
    ll1 = c.log_lik(p0, y=y, nparticles=300, seed=123, resample_threshold=0.5)
    ll2 = c.log_lik(p0, y=y, nparticles=300, seed=123, resample_threshold=0.5)
    assert ll1 == ll2


def test_compile_model_order2_particle_filter_zero_meas_error_jitter():
    yaml_text = """
declarations:
  name: order2_pf_hh_zero
  variables: [x]
  shocks: [e]
  parameters: [rho]
  observables: [x]
  measurement_errors: [me_x]

equations:
  model:
    - x = rho*x(-1) + e
  observables:
    x: x

calibration:
  parameters:
    rho: 0.9
  covariance:
    e: 0.01
  measurement_errors:
    me_x: 0.0
"""
    m = read_yaml(io.StringIO(yaml_text))
    p0 = m.p0()
    c = m.compile_model(order=2)

    y = np.zeros((10, 1))
    ll = c.log_lik(p0, y=y, nparticles=200, seed=0)
    assert np.isfinite(ll)
