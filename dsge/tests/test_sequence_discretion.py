"""Independent optimality checks for sequence-space discretion."""

from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import solve_discrete_are

from dsge.oc import solve_fixed_point
from dsge.sequence_discretion import (
    DiscretionConvergenceError,
    DiscretionError,
    discretionary_step,
    shift_policy_operator,
    solve_discretion,
)


def _forward_phillips(horizon, kappa=0.5, beta=0.9):
    """pi_t=kappa*x_t+beta*E_t*pi_(t+1)+u_t, baseline x=0."""
    M = np.zeros((2*horizon, horizon))
    for t in range(horizon):
        M[2*t+1, t] = 1
        for s in range(t, horizon):
            M[2*t, s] = kappa*beta**(s-t)
    return M


def _backward_state(horizon, a=0.7, impacts=(1.0,)):
    """State s_t=a*s_(t-1)+b'u_t and observable policy controls."""
    m = len(impacts)
    n = m+1
    M = np.zeros((n*horizon, m*horizon))
    for t in range(horizon):
        for s in range(t+1):
            M[n*t, m*s:m*(s+1)] = a**(t-s)*np.asarray(impacts)
        M[n*t+1:n*(t+1), m*t:m*(t+1)] = np.eye(m)
    baseline = np.zeros((horizon, n))
    baseline[:, 0] = a**np.arange(1, horizon+1)
    return M, baseline.reshape(-1)


def _path(result, M, baseline):
    instruments = np.linalg.solve(result.operator@M,
                                  -result.operator@baseline-result.offset)
    return baseline+M@instruments


class TestSequenceDiscretion(TestCase):
    def test_finite_phillips_curve_is_discretion_not_commitment(self):
        horizon, kappa, beta, penalty = 5, 0.5, 0.9, 0.3
        M = _forward_phillips(horizon, kappa, beta)
        W = np.diag([1.0, penalty])
        shocks = 0.6**np.arange(horizon)
        baseline = np.zeros((horizon, 2))
        for t in range(horizon):
            baseline[t, 0] = sum(beta**(s-t)*shocks[s]
                                 for s in range(t, horizon))
        result = solve_discretion(M, W, beta, method="finite")
        y = _path(result, M, baseline.reshape(-1)).reshape(horizon, 2)
        # The exact subgame-perfect solution has the contemporaneous targeting
        # condition kappa*pi+penalty*x=0, even though M includes anticipation.
        assert_allclose(kappa*y[:, 0]+penalty*y[:, 1], 0, atol=1e-12)
        pi_next = np.r_[y[1:, 0], 0.0]
        assert_allclose(y[:, 0], kappa*y[:, 1]+beta*pi_next+shocks,
                        atol=1e-12)
        Q = np.kron(np.diag(beta**np.arange(horizon)), W)
        u_commit = np.linalg.solve(M.T@Q@M, -M.T@Q@baseline.reshape(-1))
        y_commit = (baseline.reshape(-1)+M@u_commit).reshape(horizon, 2)
        self.assertGreater(np.max(np.abs(y-y_commit)), 0.05)

    def test_stationary_phillips_curve_and_tangent(self):
        horizon = 7
        M = _forward_phillips(horizon)
        result = solve_discretion(M, np.diag([1.0, 0.3]), 0.9,
                                  method="stationary")
        self.assertLessEqual(result.residual, 1e-9)
        baseline = np.zeros(2*horizon)
        baseline[::2] = 0.7**np.arange(horizon)
        y = _path(result, M, baseline).reshape(horizon, 2)
        assert_allclose(0.5*y[:, 0]+0.3*y[:, 1], 0, atol=1e-10)
        C = shift_policy_operator(result.operator[:-1], 2)
        assert_allclose(C@result.effective_response, 0, atol=1e-12)
        assert_allclose(result.effective_response[2:], 0, atol=1e-12)

    def test_finite_lqr_matches_independent_path_minimization(self):
        # Without forward-looking private behavior, finite commitment and
        # sequential optimization agree. Two instruments exercise block order.
        horizon = 5
        M, baseline = _backward_state(horizon, impacts=(1.0, -0.4))
        W = np.diag([1.0, 0.3, 0.7])
        gradient = np.array([-0.2, 0.0, 0.0])
        result = solve_discretion(M, W, 0.95, linear_loss=gradient,
                                  method="finite")
        weights = 0.95**np.arange(horizon)
        Q = np.kron(np.diag(weights), W)
        g = np.kron(weights, gradient)
        expected_u = np.linalg.solve(M.T@Q@M, -M.T@(Q@baseline+g))
        expected_y = baseline+M@expected_u
        assert_allclose(_path(result, M, baseline), expected_y, atol=1e-12)
        # The current deviation is stationary at the optimum and has strictly
        # positive loss curvature once future best responses are imposed.
        G = result.effective_response
        assert_allclose(G.T@(Q@expected_y+g), 0, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(G.T@Q@G)[0], 0)

    def test_affine_loss_and_multiple_instruments_stationary(self):
        horizon = 4
        impact = np.array([[1.0, -0.2], [0.0, 1.0], [1.0, 0.0]])
        M = np.kron(np.eye(horizon), impact)
        W = np.diag([1.0, 0.3, 0.7])
        g = np.array([-0.4, 0.2, 0.0])
        baseline = np.linspace(-0.3, 0.5, 3*horizon)
        result = solve_discretion(M, W, 0.8, linear_loss=g)
        expected = []
        for b in baseline.reshape(horizon, 3):
            u = np.linalg.solve(impact.T@W@impact, -impact.T@(W@b+g))
            expected.extend(b+impact@u)
        assert_allclose(_path(result, M, baseline), expected, atol=1e-12)

    def test_single_period(self):
        M = np.array([[1.0], [-0.5]])
        for method in ("finite", "stationary"):
            result = solve_discretion(M, np.eye(2), 1.0,
                                      linear_loss=np.array([0.2, 0.0]),
                                      method=method)
            y = _path(result, M, np.array([1.0, 0.0]))
            assert_allclose(M.T@(y+np.array([0.2, 0.0])), 0, atol=1e-12)

    def test_large_affine_target_cannot_hide_policy_nonconvergence(self):
        horizon = 5
        phillips = _forward_phillips(horizon)
        M = np.zeros((3*horizon, 2*horizon))
        for t in range(horizon):
            M[3*t, 2*t] = 1.0
            M[3*t+1:3*t+3, 1::2] = phillips[2*t:2*t+2]
        result = solve_discretion(
            M, np.diag([1.0, 1.0, 0.3]), 0.9,
            linear_loss=np.array([-1e12, 0.0, 0.0]),
        )
        baseline = np.zeros((horizon, 3))
        baseline[:, 1] = 0.7**np.arange(horizon)
        y = _path(result, M, baseline.reshape(-1)).reshape(horizon, 3)
        assert_allclose(0.5*y[:, 1]+0.3*y[:, 2], 0, atol=1e-10)
        assert_allclose(y[:, 0], 1e12, rtol=1e-12)

    def test_shift_terminal_closures(self):
        D = np.arange(6.0).reshape(1, 6)
        assert_allclose(shift_policy_operator(D, 2, "zero"),
                        [[0, 0, 0, 1, 2, 3]])
        assert_allclose(shift_policy_operator(D, 2, "repeat"),
                        [[0, 0, 0, 1, 6, 8]])

    def test_failure_diagnostics(self):
        M = _forward_phillips(5)
        with self.assertRaises(DiscretionConvergenceError):
            solve_discretion(M, np.diag([1.0, 0.3]), 0.9, max_iter=1)
        with self.assertRaisesRegex(DiscretionError, "Future-policy"):
            discretionary_step(np.eye(2), np.eye(2), np.zeros((1, 2)))
        with self.assertRaisesRegex(DiscretionError, "curvature"):
            solve_discretion(np.zeros((2, 1)), np.eye(2), 0.9,
                              method="finite")
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            solve_discretion(M, np.diag([1.0, -0.1]), 0.9)
        with self.assertRaisesRegex(ValueError, "damping"):
            solve_discretion(M, np.eye(2), 0.9, damping=0)

    def test_dennis_nonsymmetric_constraint_transpose(self):
        # D*y=e1*i+e2*nu with D=[[1,1],[0,1]], loss y'y+i^2.
        # Thus y=(i-nu,nu); at nu=1 the exact optimum is i=1/2.
        # Using D^-1 rather than D^-T in the cross term returns i=1.
        A0 = np.array([[1.0, 1.0], [0.0, 1.0]])
        zero = np.zeros((2, 2))
        A3 = np.array([[1.0], [0.0]])
        A4 = np.zeros((2, 1))
        A5 = np.array([[0.0], [1.0]])
        H1, H2, F1, F2 = solve_fixed_point(
            A0, zero, zero, A3, A4, A5, np.eye(2), np.eye(1),
            0.9, alpha=1.0,
        )
        assert_allclose(F2, [[0.5]], atol=1e-12)
        assert_allclose(H2, [[-0.5], [1.0]], atol=1e-12)
        assert_allclose(H1, 0, atol=1e-12)
        assert_allclose(F1, 0, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "alpha"):
            solve_fixed_point(A0, zero, zero, A3, A4, A5, np.eye(2),
                              np.eye(1), 0.9, alpha=0)

    def test_dennis_state_feedback_matches_independent_riccati_solution(self):
        # A second nonsymmetric test exercises the lagged-state policy F1,
        # rather than only the contemporaneous-shock policy F2 above.
        A0 = np.array([[1.0, 0.5], [0.0, 1.0]])
        A1 = 0.3*np.eye(2)
        A2 = np.zeros((2, 2))
        A3 = np.array([[1.0], [0.2]])
        A4 = np.zeros((2, 1))
        A5 = np.array([[0.0], [1.0]])
        W, Q, beta = np.diag([1.0, 2.0]), np.array([[0.4]]), 0.9
        A, B = np.linalg.solve(A0, A1), np.linalg.solve(A0, A3)
        # The old state s has stage cost (A*s+B*i)'W(A*s+B*i)+i'Q*i.
        S = solve_discrete_are(
            np.sqrt(beta)*A, np.sqrt(beta)*B, A.T@W@A,
            Q+B.T@W@B, np.eye(2), A.T@W@B,
        )
        P = W+beta*S
        expected = -np.linalg.solve(Q+B.T@P@B, B.T@P@A)
        _, _, F1, _ = solve_fixed_point(
            A0, A1, A2, A3, A4, A5, W, Q, beta, alpha=0.8,
        )
        assert_allclose(F1, expected, rtol=1e-9, atol=1e-11)
