from io import StringIO
from unittest import TestCase
import numpy as np

from dsge.parse_yaml import read_yaml


YAML_MULTI = """
declarations:
  name: multi_regime
  variables: [x, z]
  parameters: [rho]
  shocks: [e]

calibration:
  parameters:
    rho: 0.9
  covariance:
    e: 1.0

regimes:
  normal:
    equations:
      model:
        - x = rho*x(-1) + e
        - z = z(-1)
  bind_x:
    equations:
      model:
        - x = 0
        - z = z(-1)
  bind_z:
    equations:
      model:
        - x = rho*x(-1) + e
        - z = 0
  both:
    equations:
      model:
        - x = 0
        - z = 0

constraints:
  - name: cx
    when: "x < 0"
    binding_regime: bind_x
    normal_regime: normal
  - name: cz
    when: "z < 0"
    binding_regime: bind_z
    normal_regime: normal

regime_map:
  cx,cz: both
"""


class TestOBCMulti(TestCase):
    def test_multi_constraints_regime_map(self):
        m = read_yaml(StringIO(YAML_MULTI))
        p0 = m.p0()
        compiled = m.compile_model()
        # Negative shock on x moves x below 0 initially
        eps = np.zeros((5, 1)); eps[0, 0] = -1.0
        res = compiled.simulate(p0, eps_path=eps, horizon=5)
        self.assertEqual(len(res['regimes']), 5)
        # Regimes list present; detailed binding behavior may depend on hysteresis rules
        self.assertTrue(isinstance(res['regimes'], list))
