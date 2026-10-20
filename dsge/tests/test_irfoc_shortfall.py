"""Economic benchmarks for one-sided quadratic policy objectives.

The expected allocations come from scalar first-order conditions or backward
induction in the private NK equations, independently of the auxiliary-variable
representation used by the sequence solver.
"""

from functools import lru_cache
from io import StringIO
from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from dsge import read_yaml
from dsge.irfoc import IRFOC


KAPPA = 0.2
BETA = 0.99
OUTPUT_WEIGHT = 0.5
STATIC_YAML = """
declarations:
  name: one_sided_loss_benchmark
  variables: [pi, y, i, u, r]
  parameters: [kappa, gamma, beta, output_weight, target]
  shocks: [eu, er, em]
equations:
  - pi = kappa*y + u
  - y = -i + r
  - i = gamma*pi + em
  - u = eu
  - r = er
calibration:
  parameters:
    kappa: 0.2
    gamma: 1.5
    beta: 0.99
    output_weight: 0.5
    target: 0.3
"""


@lru_cache(maxsize=None)
def _model(dynamic=False):
    source = STATIC_YAML
    if dynamic:
        source = source.replace(
            "pi = kappa*y + u", "pi = beta*pi(+1) + kappa*y + u"
        ).replace("y = -i + r", "y = y(+1) - (i - pi(+1) - r)")
    model = read_yaml(StringIO(source))
    return model, model.compile_model()


def _problem(cost_push=(1.0, -1.0, 0.0, 0.4, -0.4), natural_rate=None,
             *, dynamic=False):
    """Build a reference-rule forecast directly from the private equations."""
    cost_push = np.asarray(cost_push, dtype=float)
    natural_rate = (np.zeros_like(cost_push) if natural_rate is None
                    else np.broadcast_to(natural_rate, cost_push.shape))
    periods = len(cost_push)
    reference = np.zeros((periods + 1, 3))
    for date in range(periods - 1, -1, -1):
        pi_next, y_next = reference[date + 1, :2] if dynamic else (0.0, 0.0)
        demand = y_next + pi_next + natural_rate[date]
        pi = (BETA*pi_next + KAPPA*demand + cost_push[date]) / (1 + 1.5*KAPPA)
        reference[date] = pi, demand - 1.5*pi, 1.5*pi
    baseline = pd.DataFrame(
        np.column_stack((reference[:-1], cost_push, natural_rate)),
        columns=["pi", "y", "i", "u", "r"],
    )
    model, compiled = _model(dynamic)
    return IRFOC(model, baseline, "em", compiled_model=compiled)


def _solver(problem, regime):
    return (problem.simulate_optimal_control if regime == "commitment"
            else problem.simulate_optimal_discretion)


def _shortfall_output(cost_push, threshold=0.0):
    cost_push = np.asarray(cost_push)
    unconstrained = -cost_push / KAPPA
    penalized = (-KAPPA*cost_push + OUTPUT_WEIGHT*threshold) / (
        KAPPA**2 + OUTPUT_WEIGHT
    )
    return np.where(unconstrained < threshold, penalized, unconstrained)


def _check_public_result(problem, result):
    """Auxiliary loss variables must not masquerade as physical outcomes."""
    assert list(result.simulation.columns) == list(problem.baseline.columns)
    assert list(result.shocks.columns) == list(problem.instrument_shocks)
    assert result.simulation.index.equals(problem.baseline.index)
    assert result.shocks.shape == (len(problem.baseline), len(problem.instrument_shocks))
    reconstructed = problem.baseline.to_numpy().reshape(-1) + (
        problem.M @ result.shocks.to_numpy().T.reshape(-1)
    )
    assert_allclose(reconstructed.reshape(problem.baseline.shape),
                    result.simulation, atol=2e-8, rtol=0.0)
    assert result.diagnostics["residual"] < 1e-7
    assert result.slack.to_numpy().min() >= -2e-8
    assert result.multipliers.to_numpy().min() >= -2e-8
    assert_allclose(result.slack.to_numpy()*result.multipliers.to_numpy(),
                    0.0, atol=2e-8, rtol=0.0)


@pytest.mark.parametrize("regime", ["commitment", "discretion"])
def test_shortfall_loss_matches_static_solution_on_both_sides_and_boundary(regime):
    problem = _problem()
    result = _solver(problem, regime)(
        "pi**2 + output_weight*min(y, 0)**2", discount="beta", return_details=True
    )
    expected_y = _shortfall_output(problem.baseline["u"])
    assert_allclose(result.simulation["y"], expected_y, atol=2e-8, rtol=0.0)
    assert_allclose(result.simulation["pi"],
                    KAPPA*expected_y + problem.baseline["u"], atol=2e-8)
    assert result.simulation.loc[0, "y"] < 0.0
    assert result.simulation.loc[1, "y"] > 0.0
    assert result.simulation.loc[2, "y"] == pytest.approx(0.0, abs=2e-8)
    assert_allclose(result.diagnostics["loss_auxiliaries"].to_numpy()[:, 0],
                    np.minimum(expected_y, 0.0), atol=2e-8, rtol=0.0)
    _check_public_result(problem, result)


@pytest.mark.parametrize("regime", ["commitment", "discretion"])
@pytest.mark.parametrize("notation", ["min(y - target, 0)", "Min(0, y - target)"])
def test_shortfall_target_and_calibrated_weight(regime, notation):
    problem = _problem(cost_push=(1.0, -1.0, -KAPPA*0.3, 0.0))
    simulation = _solver(problem, regime)(
        f"pi**2 + output_weight*{notation}**2", discount="beta"
    )
    expected_y = _shortfall_output(problem.baseline["u"], threshold=0.3)
    assert_allclose(simulation["y"], expected_y, atol=2e-8, rtol=0.0)
    assert simulation.loc[2, "y"] == pytest.approx(0.3, abs=2e-8)


@pytest.mark.parametrize("regime", ["commitment", "discretion"])
def test_overshoot_penalty_is_reflected_shortfall_solution(regime):
    problem = _problem()
    simulation = _solver(problem, regime)(
        "pi**2 + 0.5*max(y, 0)**2", discount="beta"
    )
    expected_y = -_shortfall_output(-problem.baseline["u"])
    assert_allclose(simulation["y"], expected_y, atol=2e-8, rtol=0.0)


@pytest.mark.parametrize("regime", ["commitment", "discretion"])
def test_two_sided_identity_matches_original_quadratic_loss(regime):
    problem = _problem()
    solve = _solver(problem, regime)
    symmetric = solve("pi**2 + 0.5*y**2", discount="beta")
    one_sided = solve(
        "pi**2 + 0.5*min(y, 0)**2 + 0.5*max(y, 0)**2", discount="beta"
    )
    assert_allclose(one_sided, symmetric, atol=2e-8, rtol=0.0)


def _several_hinges_optimum(cost_push):
    """Enumerate the scalar objective's four quadratic regions analytically."""
    thresholds, weights = (0.3, -0.2), (0.5, 0.2)
    candidates = list(thresholds)
    for shortfall, overshoot in product((False, True), repeat=2):
        active = np.array([shortfall, overshoot])
        y = (-KAPPA*cost_push + sum(
            weight*threshold for weight, threshold, include
            in zip(weights, thresholds, active) if include
        )) / (KAPPA**2 + np.dot(weights, active))
        if ((y <= thresholds[0] if shortfall else y >= thresholds[0])
                and (y >= thresholds[1] if overshoot else y <= thresholds[1])):
            candidates.append(y)

    def loss(y):
        return ((KAPPA*y + cost_push)**2
                + 0.5*min(y - 0.3, 0)**2 + 0.2*max(y + 0.2, 0)**2)

    return min(candidates, key=loss)


@pytest.mark.parametrize("regime", ["commitment", "discretion"])
def test_multiple_hinge_terms_with_overlapping_penalties(regime):
    problem = _problem()
    result = _solver(problem, regime)(
        "pi**2 + 0.5*min(y - 0.3, 0)**2 + 0.2*max(y + 0.2, 0)**2",
        discount="beta", return_details=True,
    )
    expected_y = [_several_hinges_optimum(u) for u in problem.baseline["u"]]
    assert_allclose(result.simulation["y"], expected_y, atol=2e-8, rtol=0.0)
    _check_public_result(problem, result)


@pytest.mark.parametrize("regime", ["commitment", "discretion"])
def test_shortfall_loss_and_interest_rate_floor_interact(regime):
    problem = _problem(natural_rate=(-0.5, 0.5, -0.1, 0.0, 0.0))
    result = _solver(problem, regime)(
        "pi**2 + 0.5*min(y, 0)**2", discount="beta", constraints="i >= 0",
        return_details=True,
    )
    # The IS equation translates the floor into y <= r at every date.
    expected_y = np.minimum(_shortfall_output(problem.baseline["u"]),
                            problem.baseline["r"])
    assert_allclose(result.simulation["y"], expected_y, atol=2e-8, rtol=0.0)
    assert_allclose(result.simulation["i"], problem.baseline["r"] - expected_y,
                    atol=2e-8, rtol=0.0)
    assert result.simulation["i"].min() >= -2e-8
    _check_public_result(problem, result)


def _backward_nk(cost_push, floor=None):
    """Discretion has no endogenous inherited state in this NK example."""
    expected = np.zeros((len(cost_push) + 1, 3))
    for date in range(len(cost_push) - 1, -1, -1):
        pi_next, y_next = expected[date + 1, :2]
        intercept = BETA*pi_next + cost_push[date]
        y = float(_shortfall_output(intercept))
        if floor is not None:
            y = min(y, y_next + pi_next - floor)
        expected[date] = intercept + KAPPA*y, y, y_next + pi_next - y
    return expected[:-1]


@pytest.mark.parametrize("floor", [None, -0.1])
def test_dynamic_discretion_matches_backward_solution_across_loss_boundary(floor):
    periods = 24
    cost_push = 0.7**np.arange(periods) - 3*0.4**np.arange(periods)
    problem = _problem(cost_push, dynamic=True)
    constraints = None if floor is None else f"i >= {floor}"
    result = problem.simulate_optimal_discretion(
        "pi**2 + 0.5*min(y, 0)**2", discount="beta", method="finite",
        constraints=constraints, return_details=True,
    )
    expected = _backward_nk(cost_push, floor)
    assert_allclose(result.simulation[["pi", "y", "i"]], expected,
                    atol=3e-7, rtol=0.0)
    assert expected[0, 1] > 0.0
    assert expected[2, 1] < 0.0
    _check_public_result(problem, result)


def test_shortfall_discretion_converges_with_forecast_horizon():
    cost_push = 0.7**np.arange(120) - 3*0.4**np.arange(120)
    target = _backward_nk(cost_push)[:5]
    errors = []
    for periods in (8, 16, 32):
        problem = _problem(cost_push[:periods], dynamic=True)
        simulation = problem.simulate_optimal_discretion(
            "pi**2 + 0.5*min(y, 0)**2", discount="beta"
        )
        errors.append(np.max(np.abs(
            simulation[["pi", "y", "i"]].iloc[:5].to_numpy() - target
        )))
    assert errors[1] < 0.1*errors[0]
    assert errors[2] < 0.01*errors[1]
    assert errors[2] < 2e-5


def test_dynamic_commitment_matches_enumerated_structural_quadratic_regions():
    # Eight dates permit exhaustive enumeration of output's sign. The benchmark
    # uses only Phillips-curve backward substitution, not the IRFOC mapping or
    # the auxiliary loss constraints used by the implementation.
    periods = 8
    cost_push = 0.7**np.arange(periods) - 3*0.4**np.arange(periods)
    dates = np.arange(periods)
    phillips_inverse = np.triu(BETA**(dates[None, :] - dates[:, None]))
    weights = BETA**dates
    inflation_quadratic = phillips_inverse.T @ np.diag(weights) @ phillips_inverse
    gradient = KAPPA*inflation_quadratic @ cost_push
    expected_y = None
    for active in product((False, True), repeat=periods):
        hessian = KAPPA**2*inflation_quadratic + np.diag(
            OUTPUT_WEIGHT*weights*np.asarray(active)
        )
        candidate = -np.linalg.solve(hessian, gradient)
        if np.all(np.where(active, candidate <= 1e-10, candidate >= -1e-10)):
            expected_y = candidate
            break
    assert expected_y is not None, "The independent convex regional problem must have a solution."
    expected_pi = phillips_inverse @ (KAPPA*expected_y + cost_push)
    expected_i = np.r_[expected_y[1:] + expected_pi[1:], 0.0] - expected_y
    problem = _problem(cost_push, dynamic=True)
    result = problem.simulate_optimal_control(
        "pi**2 + 0.5*min(y, 0)**2", discount="beta", return_details=True
    )
    assert_allclose(result.simulation[["pi", "y", "i"]],
                    np.column_stack((expected_pi, expected_y, expected_i)),
                    atol=3e-8, rtol=0.0)
    assert expected_y.min() < -0.01
    assert expected_y.max() > 0.01
    commitment_loss = weights @ (expected_pi**2 + OUTPUT_WEIGHT*np.minimum(expected_y, 0)**2)
    discretion = _backward_nk(cost_push)
    discretionary_loss = weights @ (
        discretion[:, 0]**2 + OUTPUT_WEIGHT*np.minimum(discretion[:, 1], 0)**2
    )
    assert commitment_loss < discretionary_loss - 0.01
    _check_public_result(problem, result)


@pytest.mark.parametrize("regime", ["commitment", "discretion"])
def test_multiple_instruments_and_loss_auxiliaries_keep_original_order(regime):
    model = read_yaml(StringIO("""
declarations:
  name: two_instrument_one_sided_benchmark
  variables: [p1, y1, i1, p2, y2, i2, u]
  parameters: [beta]
  shocks: [eu, em1, em2]
equations:
  - p1 = 0.2*y1 + u
  - y1 = -i1
  - i1 = 1.5*p1 + em1
  - p2 = 0.3*y2 - 0.5*u
  - y2 = -i2
  - i2 = 1.5*p2 + em2
  - u = eu
calibration:
  parameters: {beta: 0.99}
"""))
    u = np.array([1.0, -1.0, 0.0])
    p1, p2 = u/1.3, -0.5*u/1.45
    baseline = pd.DataFrame({
        "p1": p1, "y1": -1.5*p1, "i1": 1.5*p1,
        "p2": p2, "y2": -1.5*p2, "i2": 1.5*p2, "u": u,
    })
    problem = IRFOC(model, baseline, ["em2", "em1"], compiled_model=model.compile_model())
    result = _solver(problem, regime)(
        "p1**2 + 0.5*min(y1, 0)**2 + p2**2 + 0.2*max(y2, 0)**2",
        discount="beta", return_details=True,
    )
    expected_y1 = _shortfall_output(u)
    expected_y2 = np.where(u > 0, 0.3*0.5*u/(0.3**2 + 0.2), 0.5*u/0.3)
    assert_allclose(result.simulation[["y1", "y2"]],
                    np.column_stack((expected_y1, expected_y2)), atol=2e-8, rtol=0.0)
    auxiliary = result.diagnostics["loss_auxiliaries"]
    for name in auxiliary.columns:
        expected = (np.minimum(expected_y1, 0.0) if "Min" in name
                    else np.maximum(expected_y2, 0.0))
        assert_allclose(auxiliary[name], expected, atol=2e-8, rtol=0.0)
    _check_public_result(problem, result)


@pytest.mark.parametrize("regime", ["commitment", "discretion"])
@pytest.mark.parametrize("loss", [
    "pi**2 - 0.5*min(y, 0)**2",
    "pi**2 + min(y, 0)",
    "pi**2 + min(max(y, 0), 1)**2",
    "pi**2 + min(y**2 - 1, 0)**2",
    "pi**2 + min(y, 0)*max(pi, 0)",
])
def test_unsupported_one_sided_losses_raise_clear_errors(regime, loss):
    with pytest.raises(ValueError, match="nonnegative|squared min/max|affine"):
        _solver(_problem(), regime)(loss, discount="beta")
