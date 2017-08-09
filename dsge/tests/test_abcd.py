from __future__ import division

import numpy as np
from numpy.testing import assert_equal

from unittest import TestCase

from dsge import DSGE

import pkg_resources

class TestABCD(TestCase):

    def test_abcd(self):
        model = DSGE.DSGE.read('/home/eherbst/Dropbox/code/dsge/dsge/examples/pi/pi.yaml')

        m = model.compile_model()

        A,B,C,D = m.abcd_representation([1,1.2])

        v = np.linalg.eig(A - B @ np.linalg.inv(D) @ C)
        v = np.max(np.abs(v[0]))

        self.assertAlmostEqual(1.2, v, places=6)
