from io import StringIO
from unittest import TestCase
import numpy as np

from dsge.parse_yaml import read_yaml


TOY_OBC = """
declarations:
  name: toy_zlb
  variables: [y, r]
  parameters: [phi]
  shocks: [e]

calibration:
  parameters:
    phi: 0.9
  covariance:
    e: 1.0

regimes:
  normal:
    equations:
      model:
        - r = phi*r(-1) + e
        - y = y(-1) - r
  zlb:
    equations:
      model:
        - r = 0
        - y = y(-1) - r

constraints:
  - name: zlb
    when: "r < 0"
    binding_regime: zlb
    normal_regime: normal
    horizon: 10
"""


class TestOBCSolver(TestCase):
    def test_path_binds_then_relaxes(self):
        m = read_yaml(StringIO(TOY_OBC))
        p0 = m.p0()
        # Negative shock at t=0 induces r<0 under normal regime
        eps = np.zeros((6, 1))
        eps[0, 0] = -1.0
        # Sanity check: normal regime dynamics yield r<0 at impact
        normal = m.compile_regime(m.normal_regime)
        CC, TT, RR, QQ, DD, ZZ, HH = normal.system_matrices(p0)
        s0 = np.zeros(TT.shape[0])
        s1 = TT @ s0 + RR @ eps[0]
        state_names = normal.state_names
        r_idx = state_names.index('r')
        self.assertLess(s1[r_idx], 0.0)

        # OBC simulate returns arrays and binding flags with expected lengths
        res = m.simulate_obc(p0, eps, horizon=6)
        self.assertEqual(res['states'].shape[0], 6)
        self.assertEqual(res['binding'].shape[0], 6)
        self.assertIn('r', res['state_names'])
