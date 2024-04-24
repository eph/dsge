import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal

from unittest import TestCase

from dsge import read_yaml

class TestExternal(TestCase):

    def test_simple(self):
        from dsge import read_yaml
        from io import StringIO
        simple_dsge = StringIO("""
declarations:
  name: univariate
  variables: [x]
  parameters: [rho]
  auxiliary_parameters: [gamma]      
  shocks: [e]
  external:
        names: [half_rho]
        file: /home/eherbst/Dropbox/code/dsge/dsge/tests/external.py
        
equations:
  - x = gamma*x(-1) + e


calibration:
  parameters:
    rho: 0.85
  auxiliary_parameters:
    gamma: half_rho(rho)
""")
        f = read_yaml(simple_dsge)
        mod_simple = f.compile_model()
        p0 = f.p0()
        assert_equal(-mod_simple.GAM1(p0)[0,0], p0[0]/2)


    def test_DGS(self):
        from dsge import read_yaml
        DGS = read_yaml('/home/eherbst/Dropbox/code/dsge/dsge/examples/DGS/DGS.yaml')
        DGSlin = DGS.compile_model()
        p0 = DGS.p0()
        import time
        start = time.time()
        irf = DGSlin.impulse_response(p0)
        end = time.time()
        print(f"Execution time: {end-start} seconds")
        # print(irf['ea'].round(2))
