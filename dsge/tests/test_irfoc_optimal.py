"""Public string-loss API and independently solved constrained examples."""
from io import StringIO

import numpy as np
import pytest

from dsge import read_yaml
from dsge.irfoc import IRFOC, IRFOCOptimalResult
from dsge.sequence_policy import solve_quadratic


STATIC = """
declarations:
  name: static_policy
  variables: [pi, y, i, u, r]
  parameters: [kappa, sigma, gamma, beta]
  shocks: [eu, er, em]
equations:
  - pi = kappa*y + u
  - y = -sigma*i + r
  - i = gamma*pi + em
  - u = eu
  - r = er
calibration:
  parameters: {kappa: 0.2, sigma: 1.0, gamma: 1.5, beta: 0.99}
"""


def static_problem(shock='er', sign=-1, horizon=5):
    model = read_yaml(StringIO(STATIC))
    compiled = model.compile_model()
    baseline = sign*compiled.impulse_response(model.p0(), h=horizon-1)[shock]
    return IRFOC(model, baseline, 'em', compiled_model=compiled)


@pytest.mark.parametrize('regime', ['commitment', 'discretion'])
def test_static_lower_bound_matches_analytic_solution(regime):
    problem = static_problem()
    solve = (problem.simulate_optimal_control if regime == 'commitment'
             else problem.simulate_optimal_discretion)
    result = solve('pi**2 + 0.1*y**2', discount='beta', constraints='i >= 0', return_details=True)
    assert isinstance(result, IRFOCOptimalResult)
    np.testing.assert_allclose(result.simulation.loc[0, ['pi', 'y', 'i']], [-.2, -1, 0], atol=1e-9)
    np.testing.assert_allclose(result.simulation.iloc[1:][['pi', 'y', 'i']], 0, atol=1e-9)
    # d loss / d i at i=0 equals 2*(kappa**2+lambda)=.28.
    assert result.multipliers.iloc[0, 0] == pytest.approx(.28)
    assert result.slack.to_numpy().min() >= -1e-9
    assert result.diagnostics['residual'] < 1e-8


@pytest.mark.parametrize('regime', ['commitment', 'discretion'])
def test_affine_target_and_nonbinding_constraints(regime):
    problem = static_problem(shock='eu', sign=1)
    solve = (problem.simulate_optimal_control if regime == 'commitment'
             else problem.simulate_optimal_discretion)
    free = solve('(pi - 0.1)**2 + 0.1*y**2', discount='beta')
    constrained = solve('(pi - 0.1)**2 + 0.1*y**2', discount='beta',
                        constraints=['i >= -10', 'i <= 10'])
    np.testing.assert_allclose(constrained, free, atol=1e-8)
    # Minimize (.2*y+u-.1)^2+.1*y^2 independently at each date.
    expected_y = -.2*(problem.baseline['u'].to_numpy()-.1)/.14
    np.testing.assert_allclose(free['y'], expected_y, atol=1e-8)


def test_commitment_convex_qp_kkt_against_enumerated_two_control_solution():
    # Active u1>=1, interior u2=-.5, coupled Hessian. No model machinery.
    H = np.array([[2., 1.], [1., 2.]])
    result = solve_quadratic(H, np.zeros(2), np.array([[1., 0.], [0., 1.]]), np.array([1., -1.]))
    np.testing.assert_allclose(result.choice, [1., -.5], atol=1e-10)
    np.testing.assert_allclose(result.multipliers, [1.5, 0.], atol=1e-10)


def test_invalid_loss_and_constraint_errors():
    problem = static_problem()
    with pytest.raises(ValueError, match='convex'):
        problem.simulate_optimal_control('-pi**2')
    with pytest.raises(ValueError, match='affine'):
        problem.simulate_optimal_control('pi**2+y**2', constraints='i**2 >= 1')
    with pytest.raises(ValueError, match='weak inequalities'):
        problem.simulate_optimal_control('pi**2+y**2', constraints='i > 0')
    with pytest.raises(ValueError, match='leads/lags'):
        problem.simulate_optimal_discretion('pi**2 + i(-1)**2')


def test_infeasible_constraints_raise():
    problem = static_problem()
    with pytest.raises(ValueError, match='inconsistent|failed|converge'):
        problem.simulate_optimal_control('pi**2+y**2', constraints=['i >= 1', 'i <= 0'])


def test_dynamic_bound_discretion_against_backward_nk_solution():
    text = STATIC.replace('pi = kappa*y + u', 'pi = beta*pi(+1) + kappa*y + u')
    text = text.replace('y = -sigma*i + r', 'y = y(+1) - sigma*(i-pi(+1)-r)')
    text = text.replace('r = er', 'r = 0.5*r(-1) + er')
    model = read_yaml(StringIO(text))
    compiled = model.compile_model()
    N = 35
    baseline = -compiled.impulse_response(model.p0(), h=N-1)['er']
    problem = IRFOC(model, baseline, 'em', compiled_model=compiled)
    result = problem.simulate_optimal_discretion(
        'pi**2 + 0.1*y**2', discount='beta', constraints='i >= -0.1',
        binding_horizon=12, return_details=True,
    )
    # No endogenous inherited states: each period's discretionary problem is
    # static conditional on its successor's solution, giving exact backward FOCs.
    expected = np.zeros((N+1, 3))
    for t in range(N-1, -1, -1):
        pi_next, y_next, _ = expected[t+1]
        natural_rate = -(0.5**t)
        y_star = -0.2*0.99*pi_next/(0.2**2+0.1)
        i_star = pi_next + natural_rate + y_next - y_star
        i = max(-0.1, i_star)
        y = y_next - (i-pi_next-natural_rate)
        pi = 0.99*pi_next + 0.2*y
        expected[t] = pi, y, i
    np.testing.assert_allclose(result.simulation[['pi','y','i']].iloc[:15], expected[:15], atol=2e-8)
    assert result.slack.to_numpy().min() >= -1e-8
    assert result.multipliers.to_numpy().min() >= -1e-8
    assert (result.multipliers.iloc[:3, 0] > 0).all()
    assert result.multipliers.iloc[12:, 0].abs().max() < 1e-8


def test_constrained_affine_objective_without_unconstrained_minimum():
    result = solve_quadratic(np.zeros((1, 1)), np.ones(1), np.ones((1, 1)), np.zeros(1))
    np.testing.assert_allclose(result.choice, [0.], atol=1e-9)
    np.testing.assert_allclose(result.multipliers, [1.], atol=1e-9)
    problem = static_problem()
    simulation = problem.simulate_optimal_control('i', constraints='i >= 0')
    np.testing.assert_allclose(simulation['i'], 0., atol=1e-8)


@pytest.mark.parametrize('regime', ['commitment', 'discretion'])
def test_multiple_instruments_preserve_column_order(regime):
    model = read_yaml(StringIO("""
declarations:
  name: two_instruments
  variables: [p1, y1, i1, p2, y2, i2, u]
  parameters: [beta]
  shocks: [eu, em1, em2]
equations:
  - p1 = 0.2*y1 + u
  - y1 = -i1
  - i1 = 1.5*p1 + em1
  - p2 = 0.3*y2 + 0.5*u
  - y2 = -i2
  - i2 = 1.5*p2 + em2
  - u = eu
calibration:
  parameters: {beta: 0.99}
"""))
    compiled = model.compile_model()
    baseline = compiled.impulse_response(model.p0(), h=4)['eu']
    problem = IRFOC(model, baseline, ['em2', 'em1'], compiled_model=compiled)
    solve = (problem.simulate_optimal_control if regime == 'commitment'
             else problem.simulate_optimal_discretion)
    loss = 'p1**2 + 0.1*y1**2 + p2**2 + 0.2*y2**2 + 0.3*p1*p2'
    result = solve(loss, discount='beta', return_details=True)
    K = np.diag([.2, .3])
    inflation_weight = np.array([[1., .15], [.15, 1.]])
    output_weight = np.diag([.1, .2])
    expected_output = -np.linalg.solve(K.T@inflation_weight@K+output_weight,
                                       K.T@inflation_weight@np.array([1., .5]))
    np.testing.assert_allclose(result.simulation.loc[0, ['y1','y2']], expected_output, atol=1e-9)
    np.testing.assert_allclose(result.simulation.loc[0, ['i1','i2']], -expected_output, atol=1e-9)
    reconstructed = baseline.to_numpy().reshape(-1) + problem.M @ result.shocks.to_numpy().T.reshape(-1)
    np.testing.assert_allclose(reconstructed.reshape(baseline.shape), result.simulation, atol=1e-10)


def test_qp_recovers_from_inconsistent_intermediate_binding_guess():
    # Row scaling makes the weaker constraint look more violated initially.
    # Both equalities cannot bind together, but the inequality problem is feasible.
    result = solve_quadratic(np.ones((1, 1)), np.zeros(1),
                             np.array([[100.], [1.]]), np.array([100., 2.]))
    np.testing.assert_allclose(result.choice, [2.], atol=1e-9)
    np.testing.assert_allclose(result.multipliers, [0., 2.], atol=1e-9)
