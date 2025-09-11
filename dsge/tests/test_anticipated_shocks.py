import numpy as np
from numpy.testing import assert_array_almost_equal

from unittest import TestCase

from dsge import read_yaml

class TestAnticipatedShocks(TestCase):

    def test_parse(self):
        from io import StringIO
        simple_dsge = StringIO(
            """
declarations:
  name: univariate
  variables: [x]
  parameters: [rho]
  shocks: [e]

equations:
  - x = rho*x(-1) + e(-1)


calibration:
  parameters:
    rho: 0.85            
""")
        f = read_yaml(simple_dsge)
        mod_simple = f.compile_model()
        p0 = f.p0()
        irf = mod_simple.impulse_response(p0)
        exact = np.array([0] + [0.85**h for h in range(20)])
        assert_array_almost_equal(irf['e']['x'].values, exact)
        
        simple_dsge = StringIO(
            """
declarations:
  name: univariate
  variables: [x]
  parameters: [rho]
  shocks: [e]

equations:
  - x = rho*x(-1) + e(-2)


calibration:
  parameters:
    rho: 0.85            
""")
        f = read_yaml(simple_dsge)
        mod_simple = f.compile_model()
        p0 = f.p0()
        irf = mod_simple.impulse_response(p0)
        exact = np.array([0, 0] + [0.85**h for h in range(19)])
        assert_array_almost_equal(irf['e']['x'].values, exact)

    def test_anticipated_irf(self):
        from io import StringIO
        simple_dsge = StringIO(
            """
declarations:
  name: univariate
  variables: [x]
  parameters: [rho]
  shocks: [e]

equations:
  - x = rho*x(-1) + e(-1)


calibration:
  parameters:
    rho: 0.85            
""")
        f = read_yaml(simple_dsge)
        mod_simple = f.compile_model()
        p0 = f.p0()
        irf = mod_simple.impulse_response(p0)
        
        simple_dsge = StringIO(
            """
declarations:
  name: univariate
  variables: [x]
  parameters: [rho]
  shocks: [e]

equations:
  - x = rho*x(-1) + e


calibration:
  parameters:
    rho: 0.85            
""")
        f = read_yaml(simple_dsge)
        mod_simple = f.compile_model()
        p0 = f.p0()
        ant_irf = mod_simple.anticipated_impulse_response(p0, anticipated_h=1)
        assert_array_almost_equal(irf['e']['x'].values, ant_irf['e']['x'].values)

        ant_irf = mod_simple.anticipated_impulse_response(p0, anticipated_h=2)
        simple_dsge = StringIO(
            """
declarations:
  name: univariate
  variables: [x]
  parameters: [rho]
  shocks: [e]

equations:
  - x = rho*x(-1) + e(-2)


calibration:
  parameters:
    rho: 0.85            
""")
        f = read_yaml(simple_dsge)
        mod_simple = f.compile_model()
        p0 = f.p0()
        irf = mod_simple.impulse_response(p0)
        assert_array_almost_equal(irf['e']['x'].values, ant_irf['e']['x'].values)
    def test_sw(self):
        from dsge.examples import sw

        p0 = sw.p0()
        swlin = sw.compile_model()
        import matplotlib.pyplot as plt
        swlin.anticipated_impulse_response(p0, anticipated_h=10, h=30)['em'].r.plot(); plt.show()
