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
    with pytest.raises(ValueError, match="Observable equations must be affine"):
        m.compile_model(order=2)

