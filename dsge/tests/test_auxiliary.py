
from unittest import TestCase

from dsge import read_yaml

class TestAuxiliary(TestCase):

    def test_simple(self):
        from io import StringIO
        simple_dsge = StringIO("""
declarations:
  name: univariate
  variables: [x, y]
  parameters: [rho]
  auxiliary_parameters: [gamma, zeta]      
  shocks: [e]
  external:
        names: [half_rho]
        file: /home/eherbst/Dropbox/code/dsge/dsge/tests/external.py
        
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
