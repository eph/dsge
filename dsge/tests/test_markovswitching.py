#!/usr/bin/env python3
import numpy as np

from numpy.testing import assert_array_almost_equal

from unittest import TestCase

from dsge import read_yaml

from dsge.MarkovSwitching import MarkovSwitchingModel

class TestMarkovSwitching(TestCase):

    def setUp(self):
        from io import StringIO
        simple_dsge = StringIO(
            """
declarations:
  name: 'example_dsge'
  variables: [pi, y, i, u, re, deli]
  parameters: [beta, kappa, sigma, rho, gamma_pi, gamma_y, rho_u, rho_r, p00, p11]
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
    p00: 0.9
    p11: 0.9            
""")
        self.model1 = read_yaml(simple_dsge)
        simple_dsge = StringIO(
            """
declarations:
  name: 'example_dsge'
  variables: [pi, y, i, u, re, deli]
  parameters: [beta, kappa, sigma, rho, gamma_pi, gamma_y, rho_u, rho_r, p00, p11]
  shocks: [eu, er, em]

equations:
  - pi = 0.99*pi(-1) + kappa * y + u
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
    p00: 0.9
    p11: 0.9            
""")
        self.model2 = read_yaml(simple_dsge)

    def test_solve(self):
        transition = '[[p00, 1-p00], [1-p11, p11]]'      
        ms = MarkovSwitchingModel(self.model1, self.model1, transition)
        para1 = self.model1.p0()
        [T1, T2, R1, R2] = ms.solve_LRE(para1)

        constant_coeff = self.model1.compile_model()
        TT, RR, RC = constant_coeff.solve_LRE(para1)

        TT = TT[:-2,:-2]
        RR = RR[:-2,:]

        assert_array_almost_equal(T1, TT)
        assert_array_almost_equal(T2, TT)
        assert_array_almost_equal(R1, RR)
        assert_array_almost_equal(R2, RR)
        

    def test1(self):
        ms = MarkovSwitchingModel(self.model1, self.model2, np.array([[0.5, 0.5], [0.5, 0.5]]))
        para1 = self.model1.p0()

        ms.solve_LRE(para1)
        irfs = ms.impulse_response(para1, 0)
        #irfs['er'][['y', 'pi', 'i']].plot(); plt.show()

        sim = ms.simulate(para1)
        print(sim)
