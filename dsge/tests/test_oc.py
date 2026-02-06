import numpy as np
from numpy.testing import assert_array_almost_equal

from unittest import TestCase

from dsge import read_yaml
from dsge.oc import parse_loss, write_system_in_dennis_form, compile_commitment, compile_discretion




class TestOC(TestCase):

    def test_parse(self):
        from dsge.symbols import Variable, Parameter
        endog = [Variable('x'), Variable('y')]
        exog = [Variable('z')]
        params = [Parameter('a'), Parameter('b')]

        loss = 'a*x**2 + b*y**2 + z**2 + x*y'
        W, Q = parse_loss(loss, endog, exog, params)

        W = W.subs({params[0]: 1, params[1]: 1})
        Q = Q.subs({params[0]: 1, params[1]: 1})
        
        assert_array_almost_equal(W, np.array([[2, 1], [1, 2]]))
        assert_array_almost_equal(Q, np.array([[2]]))

    def test_parse_multi_instrument_Q(self):
        from dsge.symbols import Variable, Parameter

        endog = [Variable("x"), Variable("y")]
        policy = [Variable("z"), Variable("w")]
        params = [Parameter("a"), Parameter("b")]

        loss = "a*x**2 + b*y**2 + z**2 + 3*w**2 + z*w"
        _, Q = parse_loss(loss, endog, policy, params)

        Q = Q.subs({params[0]: 1, params[1]: 1})
        assert_array_almost_equal(Q, np.array([[2, 1], [1, 6]]))

    # Note: test_A_matrices removed due to reliance on local file path; consider adding a portable fixture later.

    def test_interest_rate_smoothing(self):
        from io import StringIO
        simple_dsge = StringIO(
            """
declarations:
  name: 'example_dsge'
  variables: [pi, y, i, u, re, deli]
  parameters: [beta, kappa, sigma, rho, gamma_pi, gamma_y, rho_u, rho_r]
  shocks: [eu, er, em]

equations:
  - pi = beta * pi(+1) + kappa * y + u
  - y = y(+1) - sigma * (i - pi(+1) - re)
  - i = rho * i(-1) + (1 - rho) * (gamma_pi * pi + gamma_y * y) + em
  - u = rho_u * u(-1) + eu
  - re = rho_r * re(-1) + er
  - deli = i - i(-1)

calibration:
  parameters:
    beta: 0.99
    kappa: 0.024
    sigma: 6.25
    rho: 0.70
    gamma_pi: 1.50
    gamma_y: 0.15
    rho_u: 0.0
    rho_r: 0.50
""")

        f = read_yaml(simple_dsge)

        mod_simple = f.compile_model()
        mod_commit = compile_commitment(f, 'pi**2 + y**2 + deli**2', 'i', 'em', beta='beta')
        p0 = f.p0()

        irf = mod_commit.impulse_response(p0, h=25)["er"].loc[:, ["i", "deli"]]
        i = irf["i"].to_numpy()
        i_lag = np.r_[0.0, i[:-1]]
        assert_array_almost_equal(irf["deli"].to_numpy(), i - i_lag, decimal=8)

    def test_compile_commitment(self):
        from io import StringIO
        simple_dsge = StringIO(
            """
declarations:
  name: 'example_dsge'
  variables: [pi, y, i, u, re, deli]
  parameters: [beta, kappa, sigma, rho, gamma_pi, gamma_y, rho_u, rho_r]
  shocks: [eu, er, em]

equations:
  - pi = beta * pi(+1) + kappa * y + u
  - y = y(+1) - sigma * (i - pi(+1) - re)
  - i = rho * i(-1) + (1 - rho) * (gamma_pi * pi + gamma_y * y) + em
  - u = rho_u * u(-1) + eu
  - re = rho_r * re(-1) + er
  - deli = i - i(-1)

calibration:
  parameters:
    beta: 0.99
    kappa: 0.024
    sigma: 6.25
    rho: 0.70
    gamma_pi: 1.50
    gamma_y: 0.15
    rho_u: 0.0
    rho_r: 0.50
""")

        f = read_yaml(simple_dsge)

        mod_simple = f.compile_model()
        mod_commit = compile_commitment(f, 'pi**2 + y**2', 'i', 'em', beta='beta')
        mod_discrt = compile_discretion(f, 'pi**2 + y**2', 'i', 'em', beta='beta')

        p0 = f.p0()
        irf_simple = mod_simple.impulse_response(p0)
        irf_commit = mod_commit.impulse_response(p0)
        irf_discrt = mod_discrt.impulse_response(p0)

        for irfs in (irf_simple, irf_commit, irf_discrt):
            self.assertIn("eu", irfs)
            self.assertIn("er", irfs)
            self.assertTrue(set(["y", "pi", "i"]).issubset(irfs["eu"].columns))
            self.assertTrue(set(["y", "pi", "i"]).issubset(irfs["er"].columns))

    def test_commitment_matches_irfoc_optimal_control_with_smoothing(self):
        from io import StringIO

        import numpy as np

        from dsge.irfoc import IRFOC

        simple_dsge = StringIO(
            """
declarations:
  name: 'example_dsge'
  variables: [pi, y, i, u, re, deli]
  parameters: [beta, kappa, sigma, rho, gamma_pi, gamma_y, rho_u, rho_r]
  shocks: [eu, er, em]

equations:
  - pi = beta * pi(+1) + kappa * y + u
  - y = y(+1) - sigma * (i - pi(+1) - re)
  - i = rho * i(-1) + (1 - rho) * (gamma_pi * pi + gamma_y * y) + em
  - u = rho_u * u(-1) + eu
  - re = rho_r * re(-1) + er
  - deli = i - i(-1)

calibration:
  parameters:
    beta: 0.99
    kappa: 0.024
    sigma: 6.25
    rho: 0.70
    gamma_pi: 1.50
    gamma_y: 0.15
    rho_u: 0.0
    rho_r: 0.50
"""
        )

        f = read_yaml(simple_dsge)
        p0 = f.p0()
        h_compare = 60
        h_full = 200
        loss = "pi**2 + y**2 + deli**2"
        cols = ["pi", "y", "i", "u", "re", "deli"]

        mod_simple = f.compile_model()
        baseline = mod_simple.impulse_response(p0, h=h_full)["er"].loc[:, cols]

        irfoc = IRFOC(f, baseline=baseline, instrument_shocks="em", p0=p0, compiled_model=mod_simple)
        sim = irfoc.simulate_optimal_control(loss, discount="beta").loc[:, cols]

        mod_commit = compile_commitment(f, loss, "i", "em", beta="beta")
        commit = mod_commit.impulse_response(p0, h=h_full)["er"].loc[:, cols]

        diff = (sim.iloc[: h_compare + 1] - commit.iloc[: h_compare + 1]).to_numpy()
        self.assertLess(float(np.max(np.abs(diff))), 1e-3)

    def test_commitment_allows_instrument_penalty(self):
        from io import StringIO

        simple_dsge = StringIO(
            """
declarations:
  name: 'example_dsge'
  variables: [pi, y, i, u, re, deli]
  parameters: [beta, kappa, sigma, rho, gamma_pi, gamma_y, rho_u, rho_r]
  shocks: [eu, er, em]

equations:
  - pi = beta * pi(+1) + kappa * y + u
  - y = y(+1) - sigma * (i - pi(+1) - re)
  - i = rho * i(-1) + (1 - rho) * (gamma_pi * pi + gamma_y * y) + em
  - u = rho_u * u(-1) + eu
  - re = rho_r * re(-1) + er
  - deli = i - i(-1)

calibration:
  parameters:
    beta: 0.99
    kappa: 0.024
    sigma: 6.25
    rho: 0.70
    gamma_pi: 1.50
    gamma_y: 0.15
    rho_u: 0.0
    rho_r: 0.50
"""
        )

        f = read_yaml(simple_dsge)
        p0 = f.p0()
        mod_commit = compile_commitment(f, "pi**2 + y**2 + i**2", "i", "em", beta="beta")
        _TT, _RR, rc = mod_commit.solve_LRE(p0)
        self.assertEqual(int(rc), 1)


#    def test_compile_commitment2(self):
#        from dsge import read_yaml
#        f = read_yaml('/home/eherbst/tmp/EHL.yaml')
#        mod_simple = f.compile_model()
#        mod_commit = compile_commitment(f, 'w**2 + pi**2 + deli**2', 'i', 'epsilon_i')
#      
#        import matplotlib.pyplot as plt
#        fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
#   
#        irf_simple = mod_simple.impulse_response(f.p0())['epsilon_z'][['g','pi','deli']]
#        irf_commit = mod_commit.impulse_response(f.p0())['epsilon_z'][['g','pi','deli']]
# 
#        irf_simple.plot(ax=ax[0], subplots=True, color='C0', legend=False)
#        irf_commit.plot(ax=ax[0], subplots=True, color='C1', legend=False)
#        plt.show()
#   
