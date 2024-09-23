import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal

from unittest import TestCase

from dsge import read_yaml
from dsge.oc import parse_loss, write_system_in_dennis_form, compile_commitment, compile_discretion

import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal

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

    def test_A_matrices(self):
        from dsge import read_yaml
        f = read_yaml('/home/eherbst/tmp/dsge_example.yaml')
        A0, A1, A2, A3, A4, A5, names = write_system_in_dennis_form(f, 'i', 'em')
        pass

    def test_compile_commitment(self):
        from dsge import read_yaml
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
        p0 = f.p0()

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

    def test_A_matrices(self):
        from dsge import read_yaml
        f = read_yaml('/home/eherbst/tmp/dsge_example.yaml')
        A0, A1, A2, A3, A4, A5, names = write_system_in_dennis_form(f, 'i', 'em')
        pass

    def test_interest_rate_smoothing(self):
        from dsge import read_yaml
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

    def test_compile_commitment(self):
        from dsge import read_yaml
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
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharex=True)

        irf_simple['eu'][['y', 'pi', 'i']].plot(ax=ax[0], subplots=True, color='C0', legend=False)
        irf_commit['eu'][['y', 'pi', 'i']].plot(ax=ax[0], subplots=True, color='C1', legend=False)
        irf_discrt['eu'][['y', 'pi', 'i']].plot(ax=ax[0], subplots=True, color='C2', legend=False, linestyle='dashed')
        ax[0,0].legend(['Taylor Rule', 'Commitment', 'Discretion'])
        # Add super title in the middle above the first row of plots
        fig.text(0.5, 0.95, "IRFs to Supply Shock", ha='center', va='center', fontsize=14)
         
        # Plot and set titles for the second row of plots
        irf_simple['er'][['y', 'pi', 'i']].plot(ax=ax[1], subplots=True, color='C0', legend=False)
        irf_commit['er'][['y', 'pi', 'i']].plot(ax=ax[1], subplots=True, color='C1', legend=False)
        irf_discrt['er'][['y', 'pi', 'i']].plot(ax=ax[1], subplots=True, color='C2', legend=False, linestyle='dashed')
         
        [axi.set_title(t) for axi, t in zip(ax[0], ['Output Gap', 'Inflation', 'Interest Rate'])]
        [axi.set_title(t) for axi, t in zip(ax[1], ['Output Gap', 'Inflation', 'Interest Rate'])]
        # Add super title in the middle above the second row of plots
        fig.text(0.5, 0.51, "IRFs to Natural Rate Shock", ha='center', va='center', fontsize=14)
        fig.subplots_adjust(hspace=0.4)
        plt.show()


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
