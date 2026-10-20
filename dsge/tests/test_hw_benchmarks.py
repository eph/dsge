"""Independent economic benchmarks for sequence-space optimal discretion.

The analytical NK solution uses a cost-push innovation, not a policy shock.
In particular, its forecast path is not re-optimized as a commitment plan at
every date: future discretionary choices respond only to inherited states.
"""

from functools import lru_cache
from io import StringIO

import numpy as np
from numpy.testing import assert_allclose
import pytest

from dsge import read_yaml
from dsge.irfoc import IRFOC
from dsge.oc import compile_commitment, compile_discretion


BETA = 0.99
KAPPA = 0.2
SIGMA = 1.0  # The coefficient multiplying the real rate in the IS equation.
RHO_U = 0.7
LAMBDA = 0.1
NK_LOSS = "pi**2 + 0.1*y**2"
SMOOTHING_LOSS = NK_LOSS + " + 0.3*deli**2"
NK_COLUMNS = ["pi", "y", "i", "u", "deli"]


@lru_cache(maxsize=None)
def _nk_model(rho_i=0.3, phi_pi=1.5, phi_y=0.2):
    """Compile a determinate reference rule, keeping private equations fixed."""
    model = read_yaml(StringIO(f"""
declarations:
  name: hw_nk_benchmark
  variables: [pi, y, i, u, deli]
  parameters: [beta, kappa, sigma, rho_u, rho_i, phi_pi, phi_y]
  shocks: [eu, em]
equations:
  - pi = beta*pi(+1) + kappa*y + u
  - y = y(+1) - sigma*(i - pi(+1))
  - i = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + em
  - u = rho_u*u(-1) + eu
  - deli = i - i(-1)
calibration:
  parameters:
    beta: {BETA}
    kappa: {KAPPA}
    sigma: {SIGMA}
    rho_u: {RHO_U}
    rho_i: {rho_i}
    phi_pi: {phi_pi}
    phi_y: {phi_y}
  covariance:
    eu: 1.0
    em: 1.0
"""))
    return model, model.compile_model(), model.p0()


@lru_cache(maxsize=None)
def _nk_irfoc(periods, rho_i=0.3, phi_pi=1.5, phi_y=0.2):
    model, compiled, parameters = _nk_model(rho_i, phi_pi, phi_y)
    baseline = compiled.impulse_response(parameters, h=periods - 1)["eu"]
    return IRFOC(
        model,
        baseline.loc[:, NK_COLUMNS],
        "em",
        p0=parameters,
        compiled_model=compiled,
    )


@lru_cache(maxsize=None)
def _discretion_path(periods=80, loss=NK_LOSS, reference=(0.3, 1.5, 0.2),
                     method="stationary"):
    return _nk_irfoc(periods, *reference).simulate_optimal_discretion(
        loss, discount="beta", method=method
    )


def _analytic_nk_path(periods):
    # The discretionary target criterion is kappa*pi + lambda*y = 0.
    # Combining it with the Phillips curve and E[u(+1)] = rho_u*u gives
    # pi = lambda*u/d, y = -kappa*u/d. The IS equation then determines i.
    denominator = KAPPA**2 + LAMBDA * (1.0 - BETA * RHO_U)
    coefficients = np.array([
        LAMBDA / denominator,
        -KAPPA / denominator,
        (RHO_U * LAMBDA + (1.0 - RHO_U) * KAPPA / SIGMA)
        / denominator,
    ])
    return RHO_U ** np.arange(periods)[:, None] * coefficients


@pytest.mark.parametrize("method", ["stationary", "finite"])
def test_hw_discretion_matches_analytical_nk_solution(method):
    """Verify levels, timing, and the discretionary target criterion."""
    simulation = _discretion_path(method=method)
    assert_allclose(
        simulation.loc[:, ["pi", "y", "i"]].iloc[:20],
        _analytic_nk_path(20),
        atol=2e-6,
        rtol=0.0,
    )
    assert_allclose(
        KAPPA * simulation["pi"].iloc[:20]
        + LAMBDA * simulation["y"].iloc[:20],
        0.0,
        atol=2e-7,
    )


def test_hw_current_policy_is_a_one_deviation_optimum():
    """Hold successor policies fixed, then perturb only today's total rate."""
    simulation = _discretion_path()
    for date in (0, 3, 8):
        current = simulation.iloc[date]
        successor = simulation.iloc[date + 1]
        # There is no endogenous state in the NK allocation problem without
        # smoothing. A current deviation changes no future discretionary action.
        def current_loss(rate):
            output = successor["y"] - SIGMA * (rate - successor["pi"])
            inflation = BETA * successor["pi"] + KAPPA * output + current["u"]
            return inflation**2 + LAMBDA * output**2

        optimum_loss = current_loss(current["i"])
        for deviation in (-0.02, 0.02):
            excess = current_loss(current["i"] + deviation) - optimum_loss
            expected_excess = SIGMA**2 * (KAPPA**2 + LAMBDA) * deviation**2
            assert excess > 0.0
            assert_allclose(excess, expected_excess, atol=2e-8, rtol=0.0)


def test_hw_discretion_is_distinct_from_commitment():
    """Forward-looking private behavior makes commitment genuinely different."""
    discretion = _discretion_path()
    model, _compiled, parameters = _nk_model()
    commitment_model = compile_commitment(model, NK_LOSS, "i", "em", beta="beta")
    commitment = commitment_model.impulse_response(parameters, h=79)["eu"]
    difference = discretion[["pi", "y", "i"]].iloc[:20].to_numpy() - (
        commitment[["pi", "y", "i"]].iloc[:20].to_numpy()
    )
    assert np.max(np.abs(difference)) > 0.1
    # This compares losses over the same long impulse response, with identical
    # initial conditions and zero inherited commitment multipliers.
    weights = BETA ** np.arange(len(discretion))
    discretionary_loss = weights @ (
        discretion["pi"] ** 2 + LAMBDA * discretion["y"] ** 2
    )
    commitment_loss = weights @ (
        commitment["pi"] ** 2 + LAMBDA * commitment["y"] ** 2
    )
    assert commitment_loss < discretionary_loss - 0.01


def test_hw_discretion_converges_with_projection_horizon():
    """The early path approaches the infinite-horizon analytical solution."""
    target = _analytic_nk_path(5)
    errors = []
    for periods in (10, 20, 80):
        simulation = _discretion_path(periods)
        errors.append(np.max(np.abs(
            simulation[["pi", "y", "i"]].iloc[:5].to_numpy() - target
        )))
    assert errors[-1] < 2e-6
    assert errors[-1] < 0.05 * errors[0]
    assert errors[1] < errors[0]


def test_hw_discretion_is_invariant_to_reference_rule():
    """Different baseline paths and policy-news IRFs imply the same optimum."""
    first = _discretion_path()
    second = _discretion_path(reference=(0.6, 2.0, 0.1))
    baseline_first = _nk_irfoc(80).baseline
    baseline_second = _nk_irfoc(80, 0.6, 2.0, 0.1).baseline
    assert np.max(np.abs(
        baseline_first[["pi", "y", "i"]].to_numpy()
        - baseline_second[["pi", "y", "i"]].to_numpy()
    )) > 0.1
    assert_allclose(
        first[["pi", "y", "i"]].iloc[:20],
        second[["pi", "y", "i"]].iloc[:20],
        atol=3e-6,
        rtol=0.0,
    )


def test_hw_discretion_matches_dennis_with_inherited_rate_state():
    """Rate smoothing requires accounting for successor policy responses."""
    model, _compiled, parameters = _nk_model()
    simulation = _discretion_path(periods=100, loss=SMOOTHING_LOSS)
    dennis_model = compile_discretion(
        model, SMOOTHING_LOSS, "i", "em", beta="beta"
    )
    transition, _impact, success = dennis_model.solve_LRE(parameters)
    assert success == 1
    assert np.max(np.abs(np.linalg.eigvals(transition))) < 1.0
    benchmark = dennis_model.impulse_response(parameters, h=99)["eu"]
    assert_allclose(
        simulation[NK_COLUMNS].iloc[:25],
        benchmark[NK_COLUMNS].iloc[:25],
        atol=2e-8,
        rtol=0.0,
    )
    assert_allclose(
        simulation["deli"],
        simulation["i"] - np.r_[0.0, simulation["i"].to_numpy()[:-1]],
        atol=1e-9,
        rtol=0.0,
    )
    # The state is economically active; this must not collapse to the static
    # target criterion from the no-smoothing benchmark.
    assert abs(KAPPA * simulation["pi"].iloc[0]
               + LAMBDA * simulation["y"].iloc[0]) > 0.01


def test_hw_inherited_state_matches_scalar_riccati_solution():
    """A closed-form Bellman benchmark checks continuation losses directly."""
    persistence, rate_weight, discount = 0.8, 0.3, 0.95
    model = read_yaml(StringIO("""
declarations:
  name: hw_backward_lq_benchmark
  variables: [x, i]
  parameters: [a, phi]
  shocks: [eu, em]
equations:
  - x = a*x(-1) + i + eu
  - i = phi*x(-1) + em
calibration:
  parameters:
    a: 0.8
    phi: -0.2
  covariance:
    eu: 1.0
    em: 1.0
"""))
    compiled, parameters = model.compile_model(), model.p0()
    baseline = compiled.impulse_response(parameters, h=39)["eu"][["x", "i"]]
    simulation = IRFOC(model, baseline, "em", p0=parameters,
                       compiled_model=compiled).simulate_optimal_discretion(
        "x**2 + 0.3*i**2", discount=discount
    )
    # V(x_previous)=P*x_previous**2. The positive Riccati root satisfies
    # P=a**2*r*(1+beta*P)/(r+1+beta*P). A current action affects the state
    # inherited by all successors, so omitting continuation changes the root.
    linear = 1.0 + rate_weight - discount * persistence**2 * rate_weight
    value = (-linear + np.sqrt(
        linear**2 + 4 * discount * persistence**2 * rate_weight
    )) / (2 * discount)
    continuation_weight = 1.0 + discount * value
    gain = continuation_weight / (rate_weight + continuation_weight)
    state_response = 1.0 - gain
    decay = persistence * state_response
    expected = decay ** np.arange(20)[:, None] * np.array([state_response, -gain])
    assert_allclose(simulation.iloc[:20], expected, atol=2e-8, rtol=0.0)
    assert abs(gain - 1.0 / (1.0 + rate_weight)) > 0.01
