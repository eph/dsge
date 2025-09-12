from numpy.testing import assert_equal

from unittest import TestCase

from dsge import read_yaml
from dsge.resource_utils import resource_path
from pathlib import Path

class TestExternal(TestCase):

    def test_simple(self):
        from io import StringIO
        ext_path = Path(__file__).with_name("external.py")
        simple_dsge = StringIO(f"""
declarations:
  name: univariate
  variables: [x]
  parameters: [rho]
  auxiliary_parameters: [gamma]      
  shocks: [e]
  external:
        names: [half_rho]
        file: {ext_path}
        
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
        with resource_path('examples/DGS/DGS.yaml') as p:
            DGS = read_yaml(str(p))
        DGSlin = DGS.compile_model()
        p0 = DGS.p0()
        import time
        start = time.time()
        irf = DGSlin.impulse_response(p0)
        end = time.time()
        print(f"Execution time: {end-start} seconds")
        # print(irf['ea'].round(2))
