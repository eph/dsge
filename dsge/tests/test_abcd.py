from __future__ import division

import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge import read_yaml
try:
    from importlib.resources import files, as_file
except Exception:  # pragma: no cover
    from importlib_resources import files, as_file  # type: ignore

class TestABCD(TestCase):

    def test_abcd(self):
        with as_file(files('dsge') / 'examples' / 'pi' / 'pi.yaml') as p:
            model = read_yaml(str(p))

        m = model.compile_model()

        A,B,C,D = m.abcd_representation([1,1.2])

        v = np.linalg.eig(A - B @ np.linalg.inv(D) @ C)
        v = np.max(np.abs(v[0]))

        self.assertAlmostEqual(1.2, v, places=6)
