import copy
import io

import numpy as np
import yaml
from numpy.testing import assert_allclose

from dsge import read_yaml
from pathlib import Path


def _read_from_dict(d):
    return read_yaml(io.StringIO(yaml.safe_dump(d)))


def _system_matrices(model):
    p0 = model.p0()
    c = model.compile_model()
    CC, TT, RR, QQ, DD, ZZ, HH = c.system_matrices(p0)
    return (CC, TT, RR, QQ, DD, ZZ, HH)


def _find_row(eq_list, lhs_name: str) -> int:
    for i, eq in enumerate(eq_list):
        if getattr(eq.lhs, "name", None) == lhs_name:
            return i
    raise AssertionError(f"Could not find equation row with LHS {lhs_name!r}")


def test_fhp_k_dict_default_matches_scalar_k():
    base = yaml.safe_load(Path("dsge/examples/fhp/fhp.yaml").read_text())

    scalar = copy.deepcopy(base)
    scalar["declarations"]["k"] = 4

    spec = copy.deepcopy(base)
    spec["declarations"]["k"] = {"default": 4}

    m_scalar = _read_from_dict(scalar)
    m_spec = _read_from_dict(spec)

    out_scalar = _system_matrices(m_scalar)
    out_spec = _system_matrices(m_spec)

    for a, b in zip(out_scalar, out_spec):
        assert_allclose(a, b, rtol=0, atol=1e-10)


def test_fhp_rowwise_mixing_uses_plan_vs_terminal_rows():
    base = yaml.safe_load(Path("dsge/examples/fhp/fhp.yaml").read_text())

    spec = copy.deepcopy(base)
    spec["declarations"]["k"] = {"default": 4, "by_lhs": {"pi": 0}}
    m = _read_from_dict(spec)
    p0 = m.p0()
    c = m.compile_model()

    # Row indices are in plan equation order (dynamic plan + static).
    cycle_plan_eqs = m["equations"]["cycle"]["plan"] + m["equations"]["static"]
    trend_plan_eqs = m["equations"]["trend"]["plan"] + m["equations"]["static"]

    pi_idx = _find_row(cycle_plan_eqs, "pi")
    c_idx = _find_row(cycle_plan_eqs, "c")
    q_idx = _find_row(cycle_plan_eqs, "q")

    assert int(c.k) == 4
    assert int(c.k_cycle_row[pi_idx]) == 0
    assert int(c.k_cycle_row[c_idx]) == 4
    assert int(c.k_cycle_row[q_idx]) == 4
    assert int(c.k_trend_row[pi_idx]) == 0
    assert int(c.k_trend_row[c_idx]) == 4
    assert int(c.k_trend_row[q_idx]) == 4

    alpha0_cycle = c.alpha0_cycle(p0)
    alpha1_cycle = c.alpha1_cycle(p0)
    beta0_cycle = c.beta0_cycle(p0)
    alphaC_cycle = c.alphaC_cycle(p0)
    alphaF_cycle = c.alphaF_cycle(p0)
    alphaB_cycle = c.alphaB_cycle(p0)
    betaS_cycle = c.betaS_cycle(p0)

    alpha0_trend = c.alpha0_trend(p0)
    alpha1_trend = c.alpha1_trend(p0)
    betaV_trend = c.betaV_trend(p0)
    alphaC_trend = c.alphaC_trend(p0)
    alphaF_trend = c.alphaF_trend(p0)
    alphaB_trend = c.alphaB_trend(p0)

    # At m=1, pi row must use terminal system; c and q rows use plan system.
    m_iter = 1

    # Cycle effective matrices at m=1
    alphaC_eff = alpha0_cycle.copy()
    alphaF_eff = np.zeros_like(alphaF_cycle)
    alphaB_eff = alpha1_cycle.copy()
    betaS_eff = beta0_cycle.copy()

    plan_rows = m_iter <= c.k_cycle_row
    alphaC_eff[plan_rows, :] = alphaC_cycle[plan_rows, :]
    alphaF_eff[plan_rows, :] = alphaF_cycle[plan_rows, :]
    alphaB_eff[plan_rows, :] = alphaB_cycle[plan_rows, :]
    betaS_eff[plan_rows, :] = betaS_cycle[plan_rows, :]

    # pi terminal row
    assert_allclose(alphaC_eff[pi_idx, :], alpha0_cycle[pi_idx, :], rtol=0, atol=1e-12)
    assert_allclose(alphaF_eff[pi_idx, :], 0.0, rtol=0, atol=1e-12)
    assert_allclose(alphaB_eff[pi_idx, :], alpha1_cycle[pi_idx, :], rtol=0, atol=1e-12)
    assert_allclose(betaS_eff[pi_idx, :], beta0_cycle[pi_idx, :], rtol=0, atol=1e-12)

    # c plan row
    assert_allclose(alphaC_eff[c_idx, :], alphaC_cycle[c_idx, :], rtol=0, atol=1e-12)
    assert_allclose(alphaF_eff[c_idx, :], alphaF_cycle[c_idx, :], rtol=0, atol=1e-12)
    assert_allclose(alphaB_eff[c_idx, :], alphaB_cycle[c_idx, :], rtol=0, atol=1e-12)
    assert_allclose(betaS_eff[c_idx, :], betaS_cycle[c_idx, :], rtol=0, atol=1e-12)

    # forward-looking c and q remain forward-looking in plan equations
    assert np.any(np.abs(alphaF_cycle[c_idx, :]) > 0)
    assert np.any(np.abs(alphaF_cycle[q_idx, :]) > 0)

    # Trend effective matrices at m=1
    alphaC_eff = alpha0_trend.copy()
    alphaF_eff = np.zeros_like(alphaF_trend)
    alphaB_eff = alpha1_trend.copy()
    betaV_eff = betaV_trend.copy()

    plan_rows = m_iter <= c.k_trend_row
    alphaC_eff[plan_rows, :] = alphaC_trend[plan_rows, :]
    alphaF_eff[plan_rows, :] = alphaF_trend[plan_rows, :]
    alphaB_eff[plan_rows, :] = alphaB_trend[plan_rows, :]
    betaV_eff[plan_rows, :] = 0.0

    # pi terminal row keeps value loading; c plan row has none
    assert_allclose(betaV_eff[pi_idx, :], betaV_trend[pi_idx, :], rtol=0, atol=1e-12)
    assert_allclose(betaV_eff[c_idx, :], 0.0, rtol=0, atol=1e-12)


def _compile_horizon_pair(base, *, short: int, long: int, expectations: int = 0):
    """Compile a model where pi owns the short counter and all other rows the long one."""

    spec = copy.deepcopy(base)
    spec["declarations"]["k"] = {"default": long, "by_lhs": {"pi": short}}
    model = _read_from_dict(spec)
    compiled = model.compile_model(expectations=expectations)
    matrices = compiled.system_matrices(model.p0())
    return model, compiled, matrices


def test_fhp_unequal_positive_horizons_use_saturated_predecessor():
    """At (1, 2), every positive owner plans today against policy (0, 1)."""

    base = yaml.safe_load(Path("dsge/examples/fhp/fhp.yaml").read_text())
    model, current, _ = _compile_horizon_pair(base, short=1, long=2)
    _, predecessor, _ = _compile_horizon_pair(base, short=0, long=1)
    p0 = model.p0()

    alphaC_cycle = current.alphaC_cycle(p0)
    alphaF_cycle = current.alphaF_cycle(p0)
    alphaB_cycle = current.alphaB_cycle(p0)
    betaS_cycle = current.betaS_cycle(p0)
    cycle_coefficient = alphaC_cycle - alphaF_cycle @ predecessor.A_cycle

    cycle_state_residual = cycle_coefficient @ current.A_cycle - alphaB_cycle
    cycle_shock_residual = (
        cycle_coefficient @ current.B_cycle
        - alphaF_cycle @ predecessor.B_cycle @ current.P(p0)
        - betaS_cycle
    )

    alphaC_trend = current.alphaC_trend(p0)
    alphaF_trend = current.alphaF_trend(p0)
    alphaB_trend = current.alphaB_trend(p0)
    trend_coefficient = alphaC_trend - alphaF_trend @ predecessor.A_trend

    trend_state_residual = trend_coefficient @ current.A_trend - alphaB_trend
    trend_value_residual = (
        trend_coefficient @ current.B_trend - alphaF_trend @ predecessor.B_trend
    )

    # q, c, and pi are the dynamic rows.  Their current counters are all
    # positive at (1, 2), so every one must satisfy its planning equation.
    for residual in (
        cycle_state_residual,
        cycle_shock_residual,
        trend_state_residual,
        trend_value_residual,
    ):
        assert_allclose(residual[:3], 0.0, rtol=0, atol=1e-12)


def test_fhp_mixed_horizon_expectations_follow_saturated_history():
    """Forecast histories decay (1, 2) to (0, 1), then saturate at (0, 0)."""

    base = yaml.safe_load(Path("dsge/examples/fhp/fhp.yaml").read_text())
    model, current, matrices = _compile_horizon_pair(
        base, short=1, long=2, expectations=2
    )
    _, predecessor, _ = _compile_horizon_pair(base, short=0, long=1)
    _, terminal, _ = _compile_horizon_pair(base, short=0, long=0)

    _, TT, RR, _, _, _, _ = matrices
    p0 = model.p0()
    P = current.P(p0)
    nx = len(model["variables"])
    nv = len(model["values"])
    ns = len(model["shocks"])
    base_states = 3 * nx + nv + ns
    tilde_columns = slice(nx, 2 * nx)
    trend_columns = slice(2 * nx, 3 * nx)

    expected_cycle = predecessor.A_cycle @ current.A_cycle
    expected_cycle_shocks = (
        predecessor.A_cycle @ current.B_cycle + predecessor.B_cycle @ P
    )
    expected_trend = predecessor.A_trend @ current.A_trend

    first_rows = base_states
    first_tilde = slice(first_rows + nx, first_rows + 2 * nx)
    first_trend = slice(first_rows + 2 * nx, first_rows + 3 * nx)
    assert_allclose(TT[first_tilde, tilde_columns], expected_cycle, rtol=0, atol=1e-12)
    assert_allclose(RR[first_tilde, :], expected_cycle_shocks, rtol=0, atol=1e-12)
    assert_allclose(TT[first_trend, trend_columns], expected_trend, rtol=0, atol=1e-12)

    expected_cycle = terminal.A_cycle @ expected_cycle
    expected_cycle_shocks = (
        terminal.A_cycle @ expected_cycle_shocks
        + terminal.B_cycle @ np.linalg.matrix_power(P, 2)
    )
    expected_trend = terminal.A_trend @ expected_trend

    second_rows = base_states + 3 * nx
    second_tilde = slice(second_rows + nx, second_rows + 2 * nx)
    second_trend = slice(second_rows + 2 * nx, second_rows + 3 * nx)
    assert_allclose(TT[second_tilde, tilde_columns], expected_cycle, rtol=0, atol=1e-12)
    assert_allclose(RR[second_tilde, :], expected_cycle_shocks, rtol=0, atol=1e-12)
    assert_allclose(TT[second_trend, trend_columns], expected_trend, rtol=0, atol=1e-12)


def test_fhp_terminal_equation_order_does_not_matter():
    base = yaml.safe_load(Path("dsge/examples/fhp/fhp.yaml").read_text())

    base_k4 = copy.deepcopy(base)
    base_k4["declarations"]["k"] = 4

    perm = copy.deepcopy(base_k4)
    perm["model"]["cycle"]["terminal"] = list(reversed(perm["model"]["cycle"]["terminal"]))

    m_base = _read_from_dict(base_k4)
    m_perm = _read_from_dict(perm)

    out_base = _system_matrices(m_base)
    out_perm = _system_matrices(m_perm)

    for a, b in zip(out_base, out_perm):
        assert_allclose(a, b, rtol=0, atol=1e-10)
