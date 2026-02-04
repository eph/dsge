import io

import numpy as np
from numpy.testing import assert_allclose

from dsge.parse_yaml import read_yaml


def test_second_order_linear_forward_looking_zero_second_derivatives():
    yaml_text = """
declarations:
  name: so_linear
  variables: [x, y]
  shocks: [e]
  parameters: [rho, beta]

equations:
  model:
    - x = rho*x(-1) + e
    - y = beta*y(1) + x
  observables:
    x: x
    y: y

calibration:
  parameters:
    rho: 0.9
    beta: 0.99
  covariance:
    e: 1.0
"""
    m = read_yaml(io.StringIO(yaml_text))
    p0 = m.p0()
    sol = m.solve_second_order(p0)

    rho = 0.9
    beta = 0.99
    denom = 1.0 - beta * rho

    # One auxiliary state created for x(-1)
    assert sol.hx.shape == (1, 1)
    assert sol.hu.shape == (1, 1)
    assert sol.gx.shape[1] == 1
    assert sol.gu.shape[1] == 1

    i_x = sol.control_names.index("x")
    i_y = sol.control_names.index("y")

    assert_allclose(sol.hx[0, 0], rho, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.hu[0, 0], 1.0, rtol=1e-10, atol=1e-10)

    # Controls: x_t and y_t as functions of state x(-1) and shock e_t.
    assert_allclose(sol.gx[i_x, 0], rho, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.gu[i_x, 0], 1.0, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.gx[i_y, 0], rho / denom, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.gu[i_y, 0], 1.0 / denom, rtol=1e-10, atol=1e-10)

    # Linear model => second-order tensors and risk shifts are zero.
    assert_allclose(sol.hxx, 0.0, rtol=0.0, atol=1e-10)
    assert_allclose(sol.gxx, 0.0, rtol=0.0, atol=1e-10)
    assert_allclose(sol.hxu, 0.0, rtol=0.0, atol=1e-10)
    assert_allclose(sol.gxu, 0.0, rtol=0.0, atol=1e-10)
    assert_allclose(sol.huu, 0.0, rtol=0.0, atol=1e-10)
    assert_allclose(sol.guu, 0.0, rtol=0.0, atol=1e-10)
    assert_allclose(sol.hss, 0.0, rtol=0.0, atol=1e-10)
    assert_allclose(sol.gss, 0.0, rtol=0.0, atol=1e-10)


def test_second_order_matches_known_polynomial_solution_no_leads():
    yaml_text = """
declarations:
  name: so_poly
  variables: [x, y]
  shocks: [e]
  parameters: [a, b, c, d, exu, k, l, m, n, oxu]

equations:
  model:
    - x = a*x(-1) + b*e + c/2*x(-1)^2 + d/2*e^2 + exu*x(-1)*e
    - y = k*x(-1) + l*e + m/2*x(-1)^2 + n/2*e^2 + oxu*x(-1)*e
  observables:
    x: x
    y: y

calibration:
  parameters:
    a: 0.8
    b: 1.2
    c: 0.6
    d: -0.4
    exu: 0.25
    k: -0.3
    l: 0.5
    m: 0.1
    n: 0.2
    oxu: -0.7
  covariance:
    e: 1.0
"""
    m = read_yaml(io.StringIO(yaml_text))
    p0 = m.p0()
    sol = m.solve_second_order(p0)

    a, b, c, d, exu = 0.8, 1.2, 0.6, -0.4, 0.25
    k, l, m2, n2, oxu = -0.3, 0.5, 0.1, 0.2, -0.7

    assert sol.hx.shape == (1, 1)
    assert sol.hu.shape == (1, 1)

    i_x = sol.control_names.index("x")
    i_y = sol.control_names.index("y")

    assert_allclose(sol.hx[0, 0], a, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.hu[0, 0], b, rtol=1e-10, atol=1e-10)

    # x_t = a*x(-1) + b*e + 0.5*c*x(-1)^2 + exu*x(-1)*e + 0.5*d*e^2
    assert_allclose(sol.gx[i_x, 0], a, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.gu[i_x, 0], b, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.gxx[i_x, 0, 0], c, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.gxu[i_x, 0, 0], exu, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.guu[i_x, 0, 0], d, rtol=1e-10, atol=1e-10)

    # y_t = k*x(-1) + l*e + 0.5*m*x(-1)^2 + oxu*x(-1)*e + 0.5*n*e^2
    assert_allclose(sol.gx[i_y, 0], k, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.gu[i_y, 0], l, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.gxx[i_y, 0, 0], m2, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.gxu[i_y, 0, 0], oxu, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.guu[i_y, 0, 0], n2, rtol=1e-10, atol=1e-10)

    # No leads => no risk correction from future uncertainty in y(+1) terms.
    assert_allclose(sol.hss, 0.0, rtol=0.0, atol=1e-10)
    assert_allclose(sol.gss, 0.0, rtol=0.0, atol=1e-10)
