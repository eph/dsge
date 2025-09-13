from io import StringIO
from unittest import TestCase

from dsge.parse_yaml import read_yaml


class TestOBCParsing(TestCase):
    def test_minimal_obc_yaml(self):
        yaml_text = """
declarations:
  name: test_obc
  variables: [x]
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
  bind:
    equations:
      model:
        - x = 0 + e

constraints:
  - name: zlb
    when: "x < 0"
    binding_regime: bind
    normal_regime: normal
    horizon: 10
"""

        model = read_yaml(StringIO(yaml_text))
        # Basic structural checks
        self.assertTrue(hasattr(model, 'regimes'))
        self.assertIn('normal', model.regimes)
        self.assertIn('bind', model.regimes)

        p0 = model.p0()
        # Placeholder IRF returns normal regime IRFs
        out = model.irf_obc(p0, h=5)
        self.assertIn('irf', out)
        self.assertIn('binding', out)
        self.assertEqual(out['binding'].shape[0], 6)

