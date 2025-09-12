
from unittest import TestCase

from dsge import read_yaml
from pathlib import Path

class TestAuxiliary(TestCase):

    def test_simple(self):
        from io import StringIO
        ext_path = Path(__file__).with_name("external.py")
        simple_dsge = StringIO(f"""
declarations:
  name: univariate
  variables: [x, y]
  parameters: [rho]
  auxiliary_parameters: [gamma, zeta]      
  shocks: [e]
  external:
        names: [half_rho]
        file: {ext_path}
        
equations:
  - x = gamma*x(-1) + e
  - y = gamma*y(-1) + e

calibration:
  parameters:
    rho: 0.85
  auxiliary_parameters:
    gamma: rho**2
    zeta: gamma+3    
""")
        
        
        f = read_yaml(simple_dsge)
        c = f.compile_model()
        import inspect
        print(inspect.getsource(f.lambdify_auxiliary))
        print(inspect.getsource(c.GAM1))
